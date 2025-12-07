import torch
from torch import nn

class LIFLayer(nn.Module):
    def __init__(self, n_in, n_out, dt=1e-3, v_th=1.0, tau_mem=2e-2):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.dt = dt
        # initialize
        self.v_th = nn.Parameter(
                        torch.rand(n_out) * 0.2 + 0.2,
                        requires_grad=False
                    ) # initialize random thresholds for each of the neurons
        self.tau_mem = tau_mem

        # Randomly initialized weight set
        self.W = nn.Parameter(torch.randn(n_in, n_out))
        
        # We initialize v as None or empty buffer; we will shape it in forward
        self.register_buffer("v", None)

    def forward(self, pre_spikes):
        # pre_spikes is shape (Batch, n_in)
        batch_size = pre_spikes.shape[0]

        #Initialize membrane potential if it doesn't exist or batch size changed
        if self.v is None or self.v.shape[0] != batch_size:
            self.v = torch.zeros(batch_size, self.n_out, device=pre_spikes.device)

        #Compute Input Current
        I = pre_spikes @ self.W  # Shape: (Batch, n_out)
        
        #Compute Decay Factor
        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=pre_spikes.device))
        
        #Update Membrane Potential
        # V[t] = decay * V[t-1] + (1-decay) * I[t]
        self.v = decay * self.v + (1 - decay) * I
        
        #Check for Spikes
        post_spikes = (self.v >= self.v_th).float()
        self.v = self.v * (1.0 - post_spikes)
        
        return post_spikes

class LIFReservoir(nn.Module):
    def __init__(self, n_in, n_reservoir, dt=1e-3, v_th=0.5, tau_mem=1e-2, sparsity=0.1, spectral_radius=0.9):
        super().__init__()
        # initialize basic parameters
        self.n_in = n_in
        self.n_reservoir = n_reservoir  
        self.dt = dt
        self.spectral_radius = spectral_radius
        self.v_th = nn.Parameter(
                        torch.rand(n_reservoir) * 0.2 + 0.2,
                        requires_grad=False
                    ) # initialize random thresholds for each of the neurons
        self.tau_mem = tau_mem
        # input to reservoir weight 
        # no grad since fixed
        self.W_in = nn.Parameter(torch.randn(n_in, n_reservoir) * 0.8, requires_grad=False)
        
        # random weights for reservoir
        W_rec = torch.randn(n_reservoir, n_reservoir)
        # make the connections sparse
        mask = (torch.rand(n_reservoir, n_reservoir) < sparsity).float()
        # apply sparsity mask
        W_rec = W_rec * mask
        # check that the weight matrix within reservoir doesn't cause explosion
        eigenvalues = torch.linalg.eigvals(W_rec)
        # check maximum eigenvalue
        max_eigenvalue = torch.max(torch.abs(eigenvalues)).item()
        if max_eigenvalue > 0:
            # ensure that the maximum eigenvalue of whole weight matrix = spectral_radius
            W_rec = W_rec * (spectral_radius / max_eigenvalue)
        
        self.W_rec = nn.Parameter(W_rec, requires_grad=False)
        # state vector
        self.register_buffer("v", None)
        self.register_buffer("I_bias", 0.05 * torch.randn(n_reservoir)) # add a slight random bias
    
    def forward(self, pre_spikes):
        batch_size = pre_spikes.shape[0]
        # initialize state vector in case spike batch changes
        if self.v is None or self.v.shape[0] != batch_size:
            self.v = torch.zeros(batch_size, self.n_reservoir, device=pre_spikes.device)
        # input current
        I_in = pre_spikes @ self.W_in
        # current within the reservoir
        I_rec = self.v @ self.W_rec
        # total current to each neuron
        I_total = I_in + I_rec
        # decay term in LIF equation
        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=pre_spikes.device))
        # apply LIF
        self.v = decay * self.v + (1 - decay) * I_total
        # check against spiking threshold
        post_spikes = (self.v >= self.v_th).float()
        self.v = self.v * (1.0 - post_spikes)
        # update state to reset to 0 if spiked
        return post_spikes
class STDPReservoir(LIFReservoir):
    def __init__(self, n_in, n_reservoir, tau_trace = 1e-3, dt=1e-3, v_th = 0.5, tau_mem = 1e-2, sparsity = 0.1, spectral_radius = 0.9):
        super().__init__(n_in, n_reservoir, dt, v_th, tau_mem, sparsity, spectral_radius)
        self.tau_trace = tau_trace
        self.register_buffer("neuron_traces", torch.zeros(n_reservoir))
    def stdp_update(self, learning_rate = 1e-4, a_plus = 0.1, a_minus = 0.12):
        W_rec = self.W_rec.data
        post_traces = self.neuron_traces.reshape(-1, 1)
        pre_traces = self.neuron_traces.reshape(1, -1)
        potentiation = a_plus * pre_traces * (1.0 - post_traces) # positive for presynaptic firing by earlier ones
        depression = -a_minus * post_traces * (1.0 - pre_traces) # positive for postysynatpc firing by later ones
        dW = learning_rate * (potentiation + depression) 
        W_rec = W_rec + dW
        W_rec = torch.clamp(W_rec, min = 0.0) # ensure something doesn't explode
        # ensure spectral radius condition still made
        eigenvalues = torch.linalg.eigvals(W_rec)
        max_eigenvalue = torch.max(torch.abs(eigenvalues)).item()
        if max_eigenvalue > 0:
            # Scale to maintain desired spectral radius
            W_rec = W_rec * (self.spectral_radius / max_eigenvalue)
        self.W_rec.data = W_rec
        return W_rec
    def forward(self, pre_spikes):
        batch_size = pre_spikes.shape[0]
        if self.v is None or self.v.shape[0] != batch_size:
            self.v = torch.zeros(batch_size, self.n_reservoir, device=pre_spikes.device)
        # input current
        I_in = pre_spikes @ self.W_in
        # current within the reservoir
        I_rec = self.v @ self.W_rec
        # total current to each neuron
        I_total = I_in + I_rec
        # decay term in LIF equation
        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=pre_spikes.device))
        # apply LIF
        self.v = decay * self.v + (1 - decay) * I_total
        # check against spiking threshold
        post_spikes = (self.v >= self.v_th).float()
        # Calculate trace decay factor
        trace_decay = torch.exp(torch.tensor(-self.dt/self.tau_trace, device=pre_spikes.device))
        # First decay existing traces
        self.neuron_traces = self.neuron_traces * trace_decay
        # Then set to 1 where neurons fired
        self.neuron_traces = torch.where(post_spikes > 0, torch.ones_like(self.neuron_traces), self.neuron_traces)
        self.v = self.v * (1.0 - post_spikes)
        self.stdp_update()
        return post_spikes

if __name__ == "__main__":
    pass
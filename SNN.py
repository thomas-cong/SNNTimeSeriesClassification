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
if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")
    
    print("\n=== Testing LIFReservoir ===")
    reservoir = LIFReservoir(n_in=1, n_reservoir=50, sparsity=0.2, spectral_radius=0.95).to(device)
    T = 200
    spike_train = torch.zeros(1, 1, device=device)
    
    print("\n--- Reservoir Simulation with Single Channel Spike Train ---")
    for t in range(T):
        if t % 20 == 0:
            spike_train[0, 0] = 1.0
        else:
            spike_train[0, 0] = 0.0
        
        reservoir_output = reservoir(spike_train)
        num_active = reservoir_output[0].sum().item()
        
        if t % 10 == 0 or spike_train[0, 0] == 1.0:
            print(f"Time {t:3d}: Input spike: {spike_train[0, 0].item():.0f} | Active neurons: {num_active:2.0f}/{reservoir.n_reservoir} | Mean voltage: {reservoir.v[0].mean().item():.3f}")
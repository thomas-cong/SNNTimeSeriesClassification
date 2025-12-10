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
                        torch.rand(n_out) * 0.1 + 0.1,
                        requires_grad=False
                    ) # initialize random thresholds for each of the neurons
        self.tau_mem = tau_mem

        # Randomly initialized weight set
        self.W = nn.Parameter(torch.randn(n_in, n_out) * 1.5)

        # We initialize v as None or empty buffer; we will shape it in forward
        self.register_buffer("v", None)

    def forward(self, pre_spikes):
        device = pre_spikes.device
        if self.v is None:
            self.v = torch.zeros(1, self.n_out, device=device)

        I = pre_spikes @ self.W

        #Compute Decay Factor
        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=pre_spikes.device))

        #Update Membrane Potential
        # V[t] = decay * V[t-1] + (1-decay) * I[t]
        self.v = decay * self.v + (1 - decay) * I

        #Check for Spikes
        post_spikes = (self.v >= self.v_th).float()
        self.v = self.v * (1.0 - post_spikes)

        return post_spikes

class STDPLayer(nn.Module):
    def __init__(self, n_pre, n_classes,
                 tau_mem=50e-3,
                 tau_trace=50e-3,
                 v_th=0.3,
                 dt=1e-2,
                 learning_rate=1e-2,
                 lateral_strength=0.05,
                 w_max=1.0,
                 w_min=-1.0):
        super().__init__()
        self.n_pre = n_pre
        self.n_classes = n_classes
        self.dt = dt
        self.tau_mem = tau_mem
        self.tau_trace = tau_trace
        self.learning_rate = learning_rate
        self.lateral_strength = lateral_strength
        self.w_max = w_max
        self.w_min = w_min

        self.W = nn.Parameter(
            torch.randn(n_pre, n_classes) * 0.2,
            requires_grad=False
        )

        self.v_th = nn.Parameter(
            torch.ones(n_classes) * v_th,
            requires_grad=False
        )

        self.register_buffer("v", None)
        self.register_buffer("eligibility", None)

    def _reset_state(self, B, device):
        self.v = torch.zeros(B, self.n_classes, device=device)
        self.eligibility = torch.zeros(B, self.n_pre, self.n_classes, device=device)

    def forward(self, pre_spikes, label=None, use_stdp=False):
        """
        pre_spikes: (B, n_pre)
        returns: post_spikes (B, n_classes)
        """
        B = pre_spikes.shape[0]
        device = pre_spikes.device

        if self.v is None:
            self._reset_state(B, device)

        # LIF membrane dynamics
        I = pre_spikes @ self.W
        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=device))
        self.v = decay * self.v + (1 - decay) * I

        raw_spikes = (self.v >= self.v_th).float()

        # Winner-take-all competition
        if raw_spikes.sum() > 1:
            winner = torch.argmax(self.v, dim=1)
            post_spikes = torch.zeros_like(raw_spikes)
            post_spikes[torch.arange(B), winner] = 1.0
        else:
            post_spikes = raw_spikes

        # Lateral inhibition
        global_activity = post_spikes.sum(dim=1, keepdim=True)
        self.v -= self.lateral_strength * global_activity

        # Eligibility trace update
        trace_decay = torch.exp(torch.tensor(-self.dt / self.tau_trace, device=device))
        self.eligibility *= trace_decay
        self.eligibility += pre_spikes.unsqueeze(2) * post_spikes.unsqueeze(1)

        # Reset membrane after spike
        self.v *= (1.0 - post_spikes)

        return post_spikes

    @torch.no_grad()
    def stdp_update(self):
        """
        Pure STDP update using eligibility traces.
        """
        # Pure STDP: use accumulated eligibility traces directly
        dW = self.learning_rate * torch.mean(self.eligibility, dim=0)
        
        self.W += dW
        self.W.clamp_(self.w_min, self.w_max)



class LIFReservoir(nn.Module):
    '''
    Class that is purely randomly initialized set of LIF neurons
    [Mainly used for STDPReservoir Base]
    '''
    def __init__(self, n_in, n_reservoir, dt=1e-3, v_th=0.3, tau_mem=1e-2, sparsity=0.1, spectral_radius=0.9):
        super().__init__()
        # initialize basic parameters
        self.n_in = n_in
        self.n_reservoir = n_reservoir
        self.dt = dt
        self.spectral_radius = spectral_radius
        self.v_th = nn.Parameter(
                        torch.rand(n_reservoir) * 0.1 + v_th,
                        requires_grad=False
                    ) # initialize random thresholds for each of the neurons
        self.tau_mem = tau_mem
        self.sparsity = sparsity #storing sparsity and spike count
        self.spectral_radius = spectral_radius
        self.register_buffer("spike_count", torch.tensor(0.0), persistent=False) # for logging number of spikes
        # input to reservoir weight
        # no grad since fixed
        W_in = torch.randn(n_in, n_reservoir) * 3.0
        self.W_in = nn.Parameter(W_in, requires_grad=False)


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
        self.register_buffer("prev_spikes", None)
        self.register_buffer("I_bias", 0.1 * torch.randn(n_reservoir) + 0.05) # add a slight random bias


    def forward(self, pre_spikes):
        device = pre_spikes.device
        if self.v is None:
            self.v = torch.zeros(1, self.n_reservoir, device=device)
            self.prev_spikes = torch.zeros(1, self.n_reservoir, device=device)
        # input current
        I_in = pre_spikes @ self.W_in
        # current within the reservoir
        I_rec = self.prev_spikes @ self.W_rec
        # total current to each neuron
        I_total = I_in + I_rec + self.I_bias
        # decay term in LIF equation
        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=pre_spikes.device))
        # apply LIF
        self.v = decay * self.v + (1 - decay) * I_total
        # check against spiking threshold
        post_spikes = (self.v >= self.v_th).float()

        self.v = self.v * (1.0 - post_spikes)
        self.spike_count += post_spikes.detach().sum() #logging number of spikes
        # update state to reset to 0 if spiked
        self.prev_spikes = post_spikes

        return post_spikes

class STDPReservoir(LIFReservoir):
    def __init__(self, n_in, n_reservoir,
                 tau_trace = 40e-3, # Adjusted to 2*tau_mem
                 dt=1e-3,
                 v_th = 0.3, 
                 tau_mem = 2e-2,
                 sparsity = 0.4,
                 spectral_radius = 1.05,
                 target_rate = 0.2,
                 eta_ip = 1e-4, # Reduced IP rate
                 lateral_strength = 0.12): # Increased inhibition
        super().__init__(n_in, n_reservoir, dt, v_th, tau_mem, sparsity, spectral_radius)
        self.tau_trace = tau_trace
        self.target_rate = target_rate
        self.eta_ip = eta_ip
        self.lateral_strength = lateral_strength
        self.register_buffer("pre_traces", torch.zeros(n_reservoir))
        self.register_buffer("post_traces", torch.zeros(n_reservoir))
        self.register_buffer("firing_rates", torch.zeros(n_reservoir))
        self.register_buffer("rate_decay", torch.tensor(0.99))
    
    def estimate_spectral_radius(self, W, n_iter=20):
        v = torch.randn(W.shape[0], device=W.device)
        for _ in range(n_iter):
            v = W @ v
            v = v / torch.norm(v)
        return torch.norm(W @ v)

    @torch.no_grad()
    def stdp_update(self, pre_spikes, post_spikes, learning_rate=1e-3, a_plus=0.1, a_minus=0.12, noise_std=1e-3):
        # pre_spikes: (N,) - from previous timestep (input to reservoir weights)
        # BUT for reservoir recurrent weights W_rec (N, N):
        # Weights are from j (pre) to i (post).
        # pre_spikes here are the reservoir spikes from previous step? 
        # Wait, the reservoir forward pass passes 'spikes' to stdp_update.
        # 'spikes' are the current output spikes.
        
        # In a recurrent layer:
        # Pre-synaptic spikes are the spikes from t-1 (prev_spikes)
        # Post-synaptic spikes are the spikes from t (spikes)
        
        W_rec = self.W_rec.clone()
        
        # Dimensions:
        # pre_traces: (1, N)
        # post_traces: (N, 1)
        # post_spikes: (N, 1)
        # pre_spikes: (1, N)
        
        post_spikes_col = post_spikes.reshape(-1, 1)
        post_spikes_row = post_spikes.reshape(1, -1)
        
        pre_traces = self.pre_traces.reshape(1, -1)
        post_traces = self.post_traces.reshape(-1, 1)
        
        pre_spikes_row = pre_spikes.reshape(1, -1)

        # Potentiation: Post fires while Pre trace is active
        # dW[i,j] (j->i) += pre_trace[j] * post_spike[i]
        potentiation = a_plus * pre_traces * post_spikes_col
        
        # Depression: Pre fires while Post trace is active
        # dW[i,j] (j->i) -= post_trace[i] * pre_spike[j]
        # pre_spikes argument passed to this function is 'spikes' (current output), 
        # but we need the actual pre-synaptic events.
        # In a recurrent net, 'pre_spikes' for the weight update are the spikes at t-1 (or the input spikes).
        # We need to pass both current and prev spikes to be precise, or just rely on traces.
        # Assuming pre_spikes arg is actually the 'pre' activity for the depression term.
        
        depression = -a_minus * post_traces * pre_spikes_row
        
        dW = learning_rate * (potentiation + depression)
        
        # weight competition: subtract mean change per postsynaptic neuron
        dW = dW - dW.mean(dim=0, keepdim=True)
        noise = noise_std * torch.randn_like(dW)
        W_rec = W_rec + dW + noise
        W_rec = W_rec.clamp(min=-0.5, max=0.5)
        rho = self.estimate_spectral_radius(W_rec)
        if rho > 0:
            W_rec *= self.spectral_radius / rho
        self.W_rec.copy_(W_rec)

    @torch.no_grad()
    def intrinsic_plasticity(self, spikes):
        self.firing_rates = self.rate_decay * self.firing_rates + (1 - self.rate_decay) * spikes
        rate_error = self.firing_rates - self.target_rate
        self.v_th.data = torch.clamp(self.v_th + self.eta_ip * rate_error, min=0.1, max=1.0)


    def forward(self, pre_spikes, use_stdp=False):
        device = pre_spikes.device
        if self.v is None:
            self.v = torch.zeros(1, self.n_reservoir, device=device)
            self.prev_spikes = torch.zeros(1, self.n_reservoir, device=device)
        
        I_in = pre_spikes @ self.W_in
        I_rec = self.prev_spikes @ self.W_rec
        I_total = I_in + I_rec
        
        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=device))
        self.v = decay * self.v + (1 - decay) * I_total
        
        post_spikes = (self.v >= self.v_th).float()
        
        global_activity = post_spikes.sum(dim=1, keepdim=True)
        self.v = self.v * (1.0 - post_spikes)
        self.v = self.v - self.lateral_strength * (global_activity / self.n_reservoir)

        
        # Update Traces
        trace_decay = torch.exp(torch.tensor(-self.dt / self.tau_trace, device=device))
        
        # Pre-trace tracks PRE-synaptic spikes (which are self.prev_spikes in a recurrent net)
        # Logic 2.1 Fixed: Use self.prev_spikes (the pre-synaptic input to this step)
        self.pre_traces = self.pre_traces * trace_decay
        self.pre_traces = torch.where(self.prev_spikes > 0, torch.ones_like(self.pre_traces), self.pre_traces)
        
        # Post-trace tracks POST-synaptic spikes (current output)
        spikes = post_spikes.squeeze(0)
        self.post_traces = self.post_traces * trace_decay
        self.post_traces = torch.where(spikes > 0, torch.ones_like(self.post_traces), self.post_traces)
        
        self.spike_count += post_spikes.detach().sum()
        
        if use_stdp:
            # Pass PRE (prev_spikes) and POST (spikes) for depression/potentiation
            self.stdp_update(self.prev_spikes.squeeze(0), spikes)
            self.intrinsic_plasticity(spikes)
            
        self.prev_spikes = post_spikes
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

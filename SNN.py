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
                 v_th=0.5,
                 dt=1e-3,
                 a_plus=0.05,
                 a_minus=0.06,
                 learning_rate=0.01,
                 lateral_strength=0.1,
                 w_max=1.0,
                 w_min=-1.0):
        super().__init__()
        self.n_pre = n_pre
        self.n_classes = n_classes
        self.dt = dt
        self.tau_mem = tau_mem
        self.tau_trace = tau_trace
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.learning_rate = learning_rate
        self.lateral_strength = lateral_strength
        self.w_max = w_max
        self.w_min = w_min

        self.W = nn.Parameter(
            torch.rand(n_pre, n_classes) * 0.5 + 0.2,
            requires_grad=False
        )
        self.v_th = nn.Parameter(
            torch.rand(n_classes) * 0.1 + 0.1,
            requires_grad=False
        )

        self.register_buffer("v", None)
        self.register_buffer("pre_trace", None)
        self.register_buffer("post_trace", None)

    @torch.no_grad()
    def stdp_update(self, pre_spikes, post_spikes, reward = 1.0):
        pre = pre_spikes.squeeze(0)
        post = post_spikes.squeeze(0)
        pre_trace = self.pre_trace.squeeze(0)
        post_trace = self.post_trace.squeeze(0)
        potentiation = self.a_plus * torch.outer(pre_trace, post)
        depression = -self.a_minus * torch.outer(pre, post_trace)
        dW = self.learning_rate * (potentiation + depression) * reward
        dW = dW - dW.mean(dim=0, keepdim=True)
        self.W.data = torch.clamp(self.W + dW, min=self.w_min, max=self.w_max)
        col_norms = self.W.norm(dim=0, keepdim=True) + 1e-8
        self.W.data = self.W / col_norms * col_norms.mean()

    def forward(self, pre_spikes, label = None, use_stdp=False):
        N_pre = pre_spikes.shape[1]
        device = pre_spikes.device
        reward = 1.0
        if self.v is None:
            self.v = torch.zeros(1, self.n_classes, device=device)
            self.pre_trace = torch.zeros(1, N_pre, device=device)
            self.post_trace = torch.zeros(1, self.n_classes, device=device)

        I = pre_spikes @ self.W
        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=device))
        self.v = decay * self.v + (1 - decay) * I
        post_spikes = (self.v >= self.v_th).float()
        if label is not None:   
            pred = post_spikes.sum(dim = 0).argmax()
            reward = 1.0 if pred == label else -1.0
        global_activity = post_spikes.sum(dim=1, keepdim=True)
        self.v = self.v - self.lateral_strength * global_activity
        trace_decay = torch.exp(torch.tensor(-self.dt / self.tau_trace, device=device))
        self.pre_trace = trace_decay * self.pre_trace + pre_spikes
        self.post_trace = trace_decay * self.post_trace + post_spikes
        if use_stdp:
            self.stdp_update(pre_spikes, post_spikes, reward = reward)
        self.v = self.v * (1.0 - post_spikes)
        return post_spikes


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
                        torch.rand(n_reservoir) * 0.1 + 0.1,
                        requires_grad=False
                    ) # initialize random thresholds for each of the neurons
        self.tau_mem = tau_mem
        self.sparsity = sparsity #storing sparsity and spike count
        self.spectral_radius = spectral_radius
        self.register_buffer("spike_count", torch.tensor(0.0), persistent=False) # for logging number of spikes
        # input to reservoir weight
        # no grad since fixed
        self.W_in = nn.Parameter(torch.randn(n_in, n_reservoir) * 1.5, requires_grad=False)


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
        device = pre_spikes.device
        if self.v is None:
            self.v = torch.zeros(1, self.n_reservoir, device=device)
        # input current
        I_in = pre_spikes @ self.W_in
        # current within the reservoir
        I_rec = self.v @ self.W_rec
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

        return post_spikes

class STDPReservoir(LIFReservoir):
    def __init__(self, n_in, n_reservoir,
                 tau_trace = 2e-2,
                 dt=1e-3,
                 v_th = 0.5, 
                 tau_mem = 1e-2,
                 sparsity = 0.2,
                 spectral_radius = 0.9,
                 target_rate = 0.1,
                 eta_ip = 1e-3,
                 lateral_strength = 0.05):
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
    def stdp_update(self, post_spikes, learning_rate=5e-2, a_plus=0.2, a_minus=0.25, noise_std=1e-3):
        W_rec = self.W_rec.clone()
        post_spikes_col = post_spikes.reshape(-1, 1)
        post_spikes_row = post_spikes.reshape(1, -1)
        pre_traces = self.pre_traces.reshape(1, -1)
        post_traces = self.post_traces.reshape(-1, 1)
        potentiation = a_plus * pre_traces * post_spikes_col
        depression = -a_minus * post_traces * post_spikes_row
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
        I_in = pre_spikes @ self.W_in
        I_rec = self.v @ self.W_rec
        I_total = I_in + I_rec
        decay = torch.exp(torch.tensor(-self.dt / self.tau_mem, device=device))
        self.v = decay * self.v + (1 - decay) * I_total
        post_spikes = (self.v >= self.v_th).float()
        global_activity = post_spikes.sum(dim=1, keepdim=True)
        self.v = self.v - self.lateral_strength * global_activity
        trace_decay = torch.exp(torch.tensor(-self.dt / self.tau_trace, device=device))
        self.pre_traces = self.post_traces * trace_decay
        spikes = post_spikes.squeeze(0)
        self.post_traces = self.post_traces * trace_decay
        self.post_traces = torch.where(spikes > 0, torch.ones_like(self.post_traces), self.post_traces)
        self.spike_count += post_spikes.detach().sum()
        if use_stdp:
            self.stdp_update(spikes)
            self.intrinsic_plasticity(spikes)
        self.v = self.v * (1.0 - post_spikes)
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

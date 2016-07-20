import numpy as np

class GSDR:
    def __init__(self, input_count, hidden_count, forced_latent_count=None, learn_rate=0.0015, sparsity=0.1, weights_init=0.01, latent_init=20.0):
        self.input_count = input_count
        self.hidden_count = hidden_count
        self.forced_latent_count = forced_latent_count
        self.learn_rate = learn_rate

        self.sdr_weights = np.random.uniform(-weights_init, weights_init, (hidden_count, input_count))
        self.sdr_bias = np.zeros(hidden_count)

        if forced_latent_count:
            self.forced_latent_weights = np.random.uniform(-latent_init, latent_init, (hidden_count, forced_latent_count))

        self.sparse_count = int(sparsity * self.hidden_count)

    def train(self, inputs, forced_latents=None):
        assert(len(inputs.shape) == 1 and inputs.shape[0] == self.input_count)
        
        # Activate SDR
        activation = self.sdr_bias + self.sdr_weights @ inputs

        # Forced Latent stuff
        if forced_latents is not None:
            assert(len(forced_latents.shape) == 1 and forced_latents.shape[0] == self.forced_latent_weights.shape[1])
            activation -= np.sum(np.square(forced_latents - self.forced_latent_weights), axis=1)

        assert(len(activation.shape) == 1 and activation.shape[0] == self.hidden_count)

        # Inhibit
        state = np.zeros_like(activation)
        sorted_indices = np.argpartition(activation, -self.sparse_count)
        state[sorted_indices[:self.sparse_count]] = 0
        state[sorted_indices[-self.sparse_count:]] = 1

        assert(np.flatnonzero(state).shape[0] == self.sparse_count)

        # Reconstruct
        reconstr = state @ self.sdr_weights

        assert(len(reconstr.shape) == 1 and reconstr.shape[0] == self.input_count)

        # Learn
        self.sdr_weights -= self.learn_rate * np.outer(state, (reconstr - inputs))

        assert(len(self.sdr_weights.shape) == 2 and self.sdr_weights.shape[0] == self.hidden_count and self.sdr_weights.shape[1] == self.input_count)

        self.sdr_bias -= self.learn_rate * activation

    def generate(self, forced_latents):
        # Forced Latent stuff
        activation = -np.sum(np.square(forced_latents - self.forced_latent_weights), axis=1)

        # Activate SDR
        activation += self.sdr_bias

        # Inhibit
        state = np.zeros_like(activation)
        sorted_indices = np.argpartition(activation, -self.sparse_count)
        state[sorted_indices[:self.sparse_count]] = 0
        state[sorted_indices[-self.sparse_count:]] = 1

        # Reconstruct
        reconstr = state @ self.sdr_weights

        return reconstr

    def generate_from_state(self, state):
        assert(state.shape == (self.hidden_count,))

        reconstr = state @ self.sdr_weights

        return reconstr

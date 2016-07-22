import numpy as np

class GSDRStack:
    def __init__(self):
        self._layers = []
        self._layers_reversed = []

    def add(self, hidden_count, input_count=None, sparsity=0.1, weights_init=0.01, forced_latent_count=None, forced_latent_init=20.0):
        self._layers.append(GSDRLayer(input_count or self._layers[-1].hidden_count, hidden_count, sparsity, weights_init, forced_latent_count, forced_latent_init))
        self._layers_reversed = list(reversed(self._layers))

    def train(self, inputs, learn_rate=0.001, forced_latents={}):
        # Calculate the SDRs
        for i, layer in enumerate(self._layers):
            layer_input = inputs if i == 0 else self._layers[i - 1].state
            layer.calculate_sdr(layer_input, forced_latents=forced_latents[i] if i in forced_latents else None)

        # Reconstruct backwards
        for layer in self._layers_reversed:
            layer.reconstruct()

        # Learn and make changes to the network
        for layer in self._layers:
            layer.learn(learn_rate)

    def generate(self, state=None, forced_latents={}):
        assert(state is not None or (len(self._layers) - 1) in forced_latents)
        assert(state is None or state.shape == self._layers[-1].state.shape)

        # Reconstruct backwards
        for i, layer in enumerate(self._layers_reversed):
            reversed_index = len(self._layers) - i - 1
            if reversed_index in forced_latents:
                layer.reconstruct_from_forced_latents(forced_latents[reversed_index])
            else:
                layer.state = state if i == 0 else self._layers_reversed[i - 1].reconstruction
                layer.reconstruct()

            # Backwards inhibition
            if i != len(self._layers) - 1:
                prev_sparse_count = self._layers_reversed[i + 1].sparse_count
                sorted_indices = np.argpartition(layer.reconstruction, -prev_sparse_count)
                layer.reconstruction[sorted_indices[:-prev_sparse_count]] = 0
                layer.reconstruction[sorted_indices[-prev_sparse_count:]] = 1

        return self._layers[0].reconstruction

class GSDRLayer:
    def __init__(self, input_count, hidden_count, sparsity, weights_init, forced_latent_count, forced_latent_init):
        self.input_count = input_count
        self.hidden_count = hidden_count
        self.forced_latent_count = forced_latent_count

        self.state = np.zeros(hidden_count)
        self.sdr_weights = np.random.uniform(-weights_init, weights_init, (hidden_count, input_count))
        self.sdr_bias = np.zeros(hidden_count)

        self.sparsity = sparsity
        self.sparse_count = int(sparsity * self.hidden_count)

        if forced_latent_count is not None:
            self.forced_latent_weights = np.random.uniform(-forced_latent_init, forced_latent_init, (hidden_count, forced_latent_count))
    
    def calculate_sdr(self, inputs, forced_latents=None):
        assert(inputs.shape == (self.input_count,))
        
        self.inputs = inputs

        # Activate SDR
        self.activation = self.sdr_bias + self.sdr_weights @ inputs

        if forced_latents is not None:
            assert(forced_latents.shape == (self.forced_latent_weights.shape[1],))
            self.activation -= np.sum(np.square(forced_latents - self.forced_latent_weights), axis=1)

        assert(self.activation.shape == (self.hidden_count,))

        # Inhibit
        sorted_indices = np.argpartition(self.activation, -self.sparse_count)
        assert(sorted_indices.shape[0] == self.hidden_count)
        assert(sorted_indices[:-self.sparse_count].shape[0] == self.hidden_count - self.sparse_count)
        assert(sorted_indices[-self.sparse_count:].shape[0] == self.sparse_count)

        self.state[sorted_indices[:-self.sparse_count]] = 0
        self.state[sorted_indices[-self.sparse_count:]] = 1

        assert(np.flatnonzero(self.state).shape[0] == self.sparse_count)

    def reconstruct(self):
        # Reconstruct
        self.reconstruction = self.state @ self.sdr_weights

        assert(self.reconstruction.shape == (self.input_count,))

    def reconstruct_from_forced_latents(self, forced_latents):
        assert(forced_latents.shape == (self.forced_latent_weights.shape[1],))

        # Activate SDR
        self.activation = self.sdr_bias - np.sum(np.square(forced_latents - self.forced_latent_weights), axis=1)
        
        assert(self.activation.shape == (self.hidden_count,))

        # Inhibit
        sorted_indices = np.argpartition(self.activation, -self.sparse_count)
        assert(sorted_indices.shape[0] == self.hidden_count)
        assert(sorted_indices[:-self.sparse_count].shape[0] == self.hidden_count - self.sparse_count)
        assert(sorted_indices[-self.sparse_count:].shape[0] == self.sparse_count)
        
        self.state[sorted_indices[:-self.sparse_count]] = 0
        self.state[sorted_indices[-self.sparse_count:]] = 1
        assert(np.flatnonzero(self.state).shape[0] == self.sparse_count)

        # Reconstruct
        self.reconstruction = self.state @ self.sdr_weights

    def learn(self, learn_rate):
        self.sdr_weights -= learn_rate * np.outer(self.state, (self.reconstruction - self.inputs))

        assert(self.sdr_weights.shape == (self.hidden_count, self.input_count))

        self.sdr_bias -= learn_rate * self.activation

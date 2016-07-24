from util import get_olivetti_faces, ImageExperiment
from gsdr import GSDRStack
import numpy as np

np.random.seed(123)

# Get the data
data, target = get_olivetti_faces()
print("Data Shape:", data.shape)
print("Target Shape:", target.shape)
input_size = (64, 64)
input_count = data.shape[1]

# Create the network
hidden_count = 256
print("Hidden count:", hidden_count)

gsdr = GSDRStack()

gsdr.add(input_count=input_count, hidden_count=hidden_count, sparsity=0.20)
gsdr.add(hidden_count=hidden_count, sparsity=0.15)
gsdr.add(hidden_count=hidden_count, sparsity=0.10)
gsdr.add(hidden_count=hidden_count, sparsity=0.05)

exp = ImageExperiment(gsdr, data, input_size, epochs=60, plot_iters=3000, learn_rate=0.003)
exp.run()

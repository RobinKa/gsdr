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
hidden_count = 200
print("Hidden count:", hidden_count)

gsdr = GSDRStack()

# No idea what the best way to stack these layers is, but let's try decreasing SDR size
# and decreasing sparsity (ie. more activated units in the SDR)
gsdr.add(input_count=input_count, hidden_count=hidden_count, sparsity=0.2)
gsdr.add(hidden_count=hidden_count, sparsity=0.15)
gsdr.add(hidden_count=hidden_count, sparsity=0.10)
gsdr.add(hidden_count=hidden_count, sparsity=0.05)
gsdr.add(hidden_count=hidden_count, sparsity=0.01)

exp = ImageExperiment(gsdr, data, input_size, epochs=100, plot_iters=1000)
exp.run()

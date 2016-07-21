from util import get_mnist, ImageExperiment
from gsdr import GSDRStack
import numpy as np

np.random.seed(123)

# Get the data
data, target = get_mnist()
print("Data Shape:", data.shape)
print("Target Shape:", target.shape)
input_size = (28, 28)
input_count = data.shape[1]

# Create the network
hidden_count = 30
print("Hidden count:", hidden_count)

gsdr = GSDRStack()

# No idea what the best way to stack these layers is, but let's try decreasing SDR size
# and decreasing sparsity (ie. more activated units in the SDR)
gsdr.add(input_count=input_count, hidden_count=hidden_count, sparsity=0.1)

exp = ImageExperiment(gsdr, data, input_size, epochs=1, plot_iters=5000, plot_count=hidden_count)
exp.run()

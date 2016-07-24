import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from util import get_mnist, ImageExperiment
from gsdr import GSDRStack

np.random.seed(123)

# Get the data
data, target = get_mnist()
print("Data Shape:", data.shape)
print("Target Shape:", target.shape)
input_size = (28, 28)
input_count = data.shape[1]

# Create the network
hidden_count = 256
print("Hidden count:", hidden_count)

forced_latent_count = 10

gsdr = GSDRStack()
gsdr.add(input_count=input_count, hidden_count=hidden_count, sparsity=0.20)
gsdr.add(hidden_count=hidden_count, sparsity=0.15)
gsdr.add(hidden_count=hidden_count, sparsity=0.10)
gsdr.add(hidden_count=hidden_count, sparsity=0.05, forced_latent_count=forced_latent_count)

last_layer_index = len(gsdr._layers)-1
digit_forced_latents = np.eye(forced_latent_count)
forced_latents = [{last_layer_index: digit_forced_latents[target[i]]} for i in range(data.shape[0])]

def plot(exp):
    f, ax = plt.subplots(2, forced_latent_count)
    
    f.set_size_inches(30, 30)

    # Generate all digits from 0 to 9
    for j in range(forced_latent_count):
        generated = exp.gsdr.generate(forced_latents={last_layer_index: digit_forced_latents[j]})
        generated = (255 * np.clip(generated, 0, 1).reshape(input_size)).astype(np.uint8)
        img = Image.fromarray(generated)
        ax[0, j].imshow(img)
        ax[0, j].set_title(str(j))
        ax[0, j].axes.get_xaxis().set_visible(False)
        ax[0, j].axes.get_yaxis().set_visible(False)

    # Interpolate between 2 and 3
    for j in range(forced_latent_count):
        latent = j / (forced_latent_count-1) * digit_forced_latents[3] + (1 - j / (forced_latent_count-1)) * digit_forced_latents[2]
        generated = exp.gsdr.generate(forced_latents={last_layer_index: latent})
        generated = (255 * np.clip(generated, 0, 1).reshape(input_size)).astype(np.uint8)
        img = Image.fromarray(generated)
        ax[1, j].imshow(img)
        ax[1, j].set_title("%.2f" % (2 + j / (forced_latent_count-1)))
        ax[1, j].axes.get_xaxis().set_visible(False)
        ax[1, j].axes.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()

exp = ImageExperiment(gsdr, data, input_size, target=target, plot_func=plot, forced_latents=forced_latents, plot_iters=5000, learn_rate=0.003, epochs=1)
exp.run()

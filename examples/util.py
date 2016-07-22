import numpy as np
from sklearn.datasets import fetch_mldata, fetch_olivetti_faces, fetch_lfw_people
import matplotlib.pyplot as plt
from PIL import Image
import math

def get_mnist():
    mnist = fetch_mldata('MNIST original')
    
    # Normalize between 0 and 1
    mnist.data = mnist.data.astype(np.float32) / 255.0
    mnist.target = mnist.target.astype(np.int32)

    return mnist.data, mnist.target

def get_olivetti_faces():
    faces = fetch_olivetti_faces()
    
    faces.data = faces.data.astype(np.float32)
    faces.target = faces.target.astype(np.int32)

    return faces.data, faces.target

def get_lfw():
    lfw = fetch_lfw_people(resize=1)
    
    lfw.data = lfw.data.astype(np.float32) / 255.0
    lfw.target = lfw.target.astype(np.int32)

    return lfw.data, lfw.target

class ImageExperiment:
    def __init__(self, gsdr, data, input_size, epochs=1, learn_rate=0.01, plot_iters=5000, plot_states=None, plot_count=20, target=None, plot_func=None, forced_latents=None):
        self.gsdr = gsdr
        self.epochs = epochs
        self.data = data
        self.target = target
        self.learn_rate = learn_rate
        self.forced_latents = forced_latents and np.array(forced_latents)
        self.plot_func = plot_func or ImageExperiment._plot
        
        self.data_count = data.shape[0]

        self.input_count = data.shape[1]
        self.input_size = input_size
        assert(self.input_size[0] * self.input_size[1] == self.input_count)
        
        self.output_size = gsdr._layers[-1].hidden_count
        self.plot_count = min(self.output_size, plot_count)
        self.plots_per_row = 10
        
        self.plot_states = plot_states
        if self.plot_states is None:
            self.plot_states = np.eye(self.output_size)[:self.plot_count]

        self.plot_iters = plot_iters
        
    def run(self):
        plot_iters = 0

        for epoch in range(self.epochs):
            print("Epoch", epoch)

            # Shuffle data
            p = np.random.permutation(len(self.data))
            self.data = self.data[p]
            if self.target is not None:
                self.target = self.target[p]
            if self.forced_latents is not None:
                self.forced_latents = self.forced_latents[p]
    
            for i in range(self.data_count):
                self.gsdr.train(self.data[i], learn_rate=self.learn_rate, forced_latents={} if self.forced_latents is None else self.forced_latents[i])

                plot_iters += 1
                if self.plot_iters is not None and plot_iters % self.plot_iters == 0:
                    print("Iteration", plot_iters)
                    self.plot_func(self)

    def _plot(exp):
        f, ax = plt.subplots(math.ceil(exp.plot_count / exp.plots_per_row), min(exp.plots_per_row, exp.plot_count))

        f.set_size_inches(30, 30)
        
        # Generate one-hot states 0 to hidden_count
        for j in range(exp.plot_count):
            a = None
            if exp.plot_count > exp.plots_per_row:
                a = ax[j // exp.plots_per_row, j % exp.plots_per_row]
            else:
                a = ax[j]

            generated = exp.gsdr.generate(exp.plot_states[j])
            generated = (255 * np.clip(generated, 0, 1).reshape(exp.input_size)).astype(np.uint8)
            img = Image.fromarray(generated)

            a.imshow(img, cmap='Greys_r')
            a.axes.get_xaxis().set_visible(False)
            a.axes.get_yaxis().set_visible(False)

        plt.tight_layout()
        plt.show()
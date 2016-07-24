# gsdr
Generative Sparse Distributed Representations, a fast generative model written in Python (Original C++ implementation https://github.com/222464/GSDR)

[Demonstration of the SDRs it learns and its reconstructions on faces](https://www.youtube.com/watch?v=m_xwH8cTMuo)

# Dependencies
- Python 3
- Python libraries
    - numpy
    - scipy
    
# Installation
`pip install gsdr` or clone and `python setup.py install`

# Usage
(More extensive examples and IPython notebooks can be found in examples/)

With labeled data:
```Python
data, labels = ...

num_labels = 10

# Data: (batches, num_features)
# Labels: (batches,) (contains numbers from 0 to num_labels-1, eg. 10 for MNIST)

# Build the GSDR network (only one layer for now)
gsdr = GSDRStack()
gsdr.add(input_count=data.shape[1], hidden_count=256, sparsity=0.1, forced_latent_count=num_labels)

forced_latents = np.eye(num_labels)

# Train once for each data point
for i in range(data.shape[0]):
    gsdr.train(data[i], forced_latents={0: forced_latents[labels[i]]})
    
# Generate one example for each label
for i in range(num_labels):
    generated = gsdr.generate(forced_latents={0: forced_latents[i]})
```

With unlabeled data:
```Python
data = ...

# Data: (batches, num_features)

# Build the GSDR network (only one layer for now)
gsdr = GSDRStack()
gsdr.add(input_count=data.shape[1], hidden_count=256, sparsity=0.1)

# Train once for each data point
for i in range(data.shape[0]):
    gsdr.train(data[i])
    
states = np.eye(hidden_count)

# Generate one example for each one-hot state
for i in range(hidden_count):
    generated = gsdr.generate(states[i])
```

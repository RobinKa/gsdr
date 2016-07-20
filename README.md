# gsdr
Generative Sparse Distributed Representations, a fast generative model written in Python (Original C++ implementation https://github.com/222464/GSDR)

# Dependencies
- Python 3
- Python libraries
    - numpy

# Usage
(More extensive examples can be found in the example scripts or the IPython notebooks)

With labeled data:
```Python
data, labels = ...

num_labels = 10

# Data: (batches, num_features)
# Labels: (batches,) (contains numbers from 0 to num_labels-1, eg. 10 for MNIST)

gsdr = GSDR(input_count=data.shape[1], hidden_count=100, forced_latent_count=labels.shape[0])

forced_latents = np.eye(labels.shape[0])

# Train once for each data point
for i in range(data.shape[1]):
    gsdr.train(data[i], forced_latents[labels[i]])
    
# Generate one example for each label
for i in range(num_labels):
    generated = gsdr.generate(forced_latents[i])
```

With unlabeled data:
```Python
data = ...
# Data: (batches, num_features)

hidden_count = 100

gsdr = GSDR(input_count=data.shape[1], hidden_count=hidden_count)



# Train once for each data point
for i in range(data.shape[1]):
    gsdr.train(data[i], forced_latents[labels[i]])
    
states = np.eye(hidden_count)
# Generate one example for each one-hot state
for i in range(hidden_count):
    generated = gsdr.generate_from_state(states[i])
```

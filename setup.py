from distutils.core import setup

setup(
    name="gsdr",
    version= "0.1.0",
    description="Generative Sparse Distributed Representations, a fast generative model",
    author="Robin Kahlow (Toraxxx)",
    author_email="xtremegosugaming@gmail.com",
    maintainer="Robin Kahlow (Toraxxx)",
    maintainer_email="xtremegosugaming@gmail.com",
    url="https://github.com/ToraxXx/gsdr",
    requires=["numpy"],
    license= "MIT",
    package_dir={"": "src"},
    packages=["gsdr"],
    platforms=["any"],
)

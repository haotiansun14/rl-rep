import os
import pathlib

from setuptools import find_packages, setup


def get_version() -> str:
    init = open(os.path.join("spectralrl", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

ROOT_DIR = pathlib.Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text()
VERSION = get_version()

def get_install_requires():
    return [
        "gym>=0.23.1,<=0.24.1",
        "tqdm",
        "h5py",
        "Cython==0.29.36"
        "imageio",
        "numpy",
        "torch",
        "tensorboard",
        "wandb",
    ]

def get_extras_require():
    return {}

setup(
    name                = "spectralrl",
    version             = VERSION,
    description         = "A library dedicated to exploring the field of Representation Learning (RepL) with a specific focus on Reinforcement Learning (RL) and Causal Inference",
    long_description    = README,
    long_description_content_type = "text/markdown",
    url                 = "https://github.com/haotiansun14/spectral-rl2",
    license             = "MIT",
    packages            = find_packages(),
    include_package_data = True,
    tests_require=["pytest", "mock"],
    python_requires=">=3.7",
    install_requires = get_install_requires()
)

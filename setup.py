from setuptools import find_packages, setup

VERSION = "0.1.0.dev"

info = dict(
    name="mgnn",
    version=VERSION,
    description="MGNN is a reimplementation of MatErials Graph Network (MEGNet) and Materials 3-body Graph Networks "
    "(M3GNet) in Deep Graph Library (DGL).",
    packages=find_packages(),
    package_data={
        "mgnn": ["*.json", "*.md"],
        "mgnn.utils": ["*.npy"],
    },
    include_package_data=True,
    install_requires=(
        "torch",
        "dgl",
    ),
)

setup(**info)

from setuptools import setup, find_packages

VERSION = '0.1.0.dev'

info = dict(
    name='megnet',
    version=VERSION,
    description='MegNet pytorch conversion, testing verion',

    packages= find_packages(),
    package_data={
        "megnet": ["*.json", "*.md"],
        "megnet.utils": ["*.npy"],
    },
    include_package_data=True,
    install_requires=(
        'torch',
        'dgl',
    ),

)

setup(**info)
from setuptools import setup, find_packages

VERSION = '0.1.0.dev'

info = dict(
    name='megnet',
    version=VERSION,
    description='MegNet pytorch conversion',

    packages=['megnet'],

    install_requires=(
        'torch',
        'dgl',
    ),

)

setup(**info)
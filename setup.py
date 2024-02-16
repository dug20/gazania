"""
Allows installation via pip, e.g. by navigating to this directory with the command prompt, and using 'pip install .'

Note that you will also have to run some form of 'conda env create -f environment.yml' to install the conda packages (such as cantera - which isn't available on pip).

Otherwise you can also just do 'conda install cantera' if you are using Anaconda.
"""

from setuptools import setup, find_packages

setup(
    name='gazania',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy', 'tabulate'],
    description='Gas turbine and jet engine cycle tools',
)
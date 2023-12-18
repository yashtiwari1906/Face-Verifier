from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='Face-Verifier-Server',
    version='1.0',
    python_requires='>=3.8',
    packages=find_packages("'Face-Verifier-Server"),
    install_requires= required
)

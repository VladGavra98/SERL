from setuptools import setup, find_packages


install_requires = []
with open("requirements.txt", "r", encoding='utf-8') as f:
    lines = f.readlines()
    install_requires = [line.rstrip() for line in lines]

with open("README.md", "r", encoding='utf-8') as f:
    readme = f.read()

setup(
    name="SERL",
    author="V.Gavra",
    version="0.1",
    long_description=readme,
    packages=find_packages(),
    install_requires=install_requires,
)
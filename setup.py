import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from codecs import open

with open("README.md", encoding="utf-8") as file:
    readme = file.read()

setup(
    name="densify",
    description="Algorithm for oversampling point clouds",
    long_description=readme,
    long_description_content_type="text/markdown",
    version="v1.0.0",
    packages=["densify"],
    python_requires=">=3",
    url="https://github.com/shobrook/densify",
    author="shobrook",
    author_email="shobrookj@gmail.com",
    # classifiers=[],
    install_requires=[],
    keywords=["oversampling", "point-cloud", "machine-learning", "regularization"],
    license="MIT"
)

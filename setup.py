from setuptools import setup, find_packages

setup(name="point-cloud-to-patches",
      author="Mohit Motwani",
      description="Package to Train a DNN to produce Coons Patch control points from point clouds",
      version="1.0.0",
      packages=find_packages(include=["src", "src.*"]),
      install_requires=["torch", "numpy", "torch-tools", "tqdm"])

from setuptools import find_packages
from distutils.core import setup

setup(name='unitree_rl_gym',
      version='1.0.0',
      author='Unitree Robotics',
      license="BSD-3-Clause",
      packages=find_packages(),
      author_email='support@unitree.com',
      description='Template RL environments for Unitree Robots',
      install_requires=['matplotlib','mujoco==3.2.3', 'pyyaml','numpy==1.22.1', 'scipy'])
      #install_requires=['matplotlib','mujoco==3.2.3', 'pyyaml','pycuda<2022.1','numpy==1.22.1','onnx'])

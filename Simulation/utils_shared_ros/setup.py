#!/usr/bin/env python3
from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['utils_shared_ros'],
    package_dir={'': 'src'},
    requires=['std_msgs', 'rospy', 'sensor_msgs']
)

setup(**setup_args)
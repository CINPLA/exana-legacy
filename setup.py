# -*- coding: utf-8 -*-
from setuptools import setup
import os

from setuptools import setup, find_packages
import versioneer

long_description = open("README.md").read()

install_requires = ['numpy', 'matplotlib']

setup(
    name="exana",
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)

# -*- coding: utf-8 -*-
from setuptools import setup
import os

from setuptools import setup, find_packages
import versioneer

long_description = open("README.md").read()

install_requires = [
    'neo>=0.5',
    'numpy>=1.9',
    'quantities>=0.10.1',
    'matplotlib']
extras_require = {
    'stuff': ['pandas>=0.14.1',
               'scipy>=0.14.0'],
    'docs': ['numpydoc>=0.5',
             'sphinx>=1.2.2']
}

setup(
    name="exana",
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(),
    include_package_data=True,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass()
)

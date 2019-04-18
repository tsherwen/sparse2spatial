from setuptools import setup, find_packages
import os

VERSION = '0.0.3'
DISTNAME = 'sparse2spatial'
DESCRIPTION = "Convert sparse spatial data to a spatially and temporally resolved 2D/3D data"
AUTHOR = 'Tomas Sherwen'
AUTHOR_EMAIL = 'tomas.sherwen@york.ac.uk'
URL = 'https://github.com/tsherwen/Sparse2Spatial'
LICENSE = 'MIT'
PYTHON_REQUIRES = '>=3.5'

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['esmpy', 'xarray', 'numpy', 'scipy']

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
]


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name=DISTNAME,
      version=VERSION,
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=readme(),
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      url=URL,
      packages=find_packages())

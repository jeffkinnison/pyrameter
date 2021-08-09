#!/usr/bin/env python
from setuptools import setup

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='pyrameter',
    version='0.2.1',
    description='Structure, sample, and savor hyperparameter searches',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jeffkinnison/pyrameter',
    author='Jeff Kinnison',
    author_email='jkinniso@nd.edu',
    packages=['pyrameter',
              'pyrameter.backend',
              'pyrameter.domains',
              'pyrameter.methods',],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: POSIX',
        'Operating System :: Unix',
    ],
    keywords='machine_learning hyperparameters',
    install_requires=[
        'dill',
        'matplotlib',
        'numpy',
        'pandas',
        'pymongo',
        'scipy',
        'scikit-learn',
        'six',
    ],
)

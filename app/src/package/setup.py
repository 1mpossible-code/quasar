from setuptools import setup, find_packages
import codecs
import os

VERSION = '0.2.2'
DESCRIPTION = 'Q.U.A.S.A.R. - Quantum Automated System for Advanced Recycling'

# Setting up
setup(
    name="quantum-automated-system-for-advanced-recycling",
    version=VERSION,
    description=DESCRIPTION,
    url='https://github.com/1mpossible-code/quasar', 
    packages=['Quasar'],

    install_requires=[
        "pennylane ~= 0.32.0",
        "tensorflow ~= 2.13.0",
        "Pillow ~= 10.0.1",
    ],
    keywords=['python', 'quasar', 'qunatum', 'convolution', 'quanvolution'],
    
    classifiers=[
       'Development Status :: 2 - Pre-Alpha',
       'Intended Audience :: Education',
       'Operating System :: OS Independent',
       'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
       'Programming Language :: Python :: 3',
       'Programming Language :: Python :: 3.5',
       'Programming Language :: Python :: 3.6',
       'Programming Language :: Python :: 3.7',
       'Programming Language :: Python :: 3.8',
       'Programming Language :: Python :: 3.9',
       'Programming Language :: Python :: 3.10',
       'Programming Language :: Python :: 3.11',
     ],
)
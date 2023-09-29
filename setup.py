from setuptools import setup, find_packages
setup(
 name="quantum-automated-system-for-advanced-recycling",
 version='0.1',
 description='Q.U.A.S.A.R. - Quantum Automated System for Advanced Recycling',
 url='https://github.com/1mpossible-code/quasar', 
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
 keywords=['python', 'quasar', 'qunatum', 'convolution', 'quanvolution'],
 packages=find_packages("app/src"),
 package_dir={"": "app/src"},
)
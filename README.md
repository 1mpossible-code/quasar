# Q.U.A.S.A.R. - Quantum Automated System for Advanced Recycling


## Table of Contents
- [Structure](#structure)
- [Setup](#setup)
- [How to run notebooks](#how-to-run-notebooks)

## Structure

- app/
  - compare/
    - *.cpp
    - *.jpg
  - src/
    - notebooks/
      - *.ipynb
    - *.cpp
  - release/ - potentially directory to hold compiled files to push for release such as executables for different systems
- requirements.txt

## Setup

```bash
# 1. Clone the repository
git clone git@github.com:1mpossible-code/quasar.git
# 2. Change directory to the repository
cd ./quasar
# 3. Create a virtual environment for python
python3 -m venv venv
# 4. Activate the virtual environment
source venv/bin/activate
# 5. Install the requirements
pip install -r requirements.txt
```

## How to run notebooks

```bash
# 1. Activate the virtual environment
source venv/bin/activate
# 2. Change directory to notebooks
cd ./app/src/notebooks
# 3. Run jupyter notebook
jupyter notebook
```
# Q.U.A.S.A.R. - Quantum Automated System for Advanced Recycling

## Table of Contents

-   [Motivation](#Motivation)
-   [Project description](#Porject-description)
-   [How it works](#How-it-works)
-   [Model Performance](#Model-Performance)
-   [Set up](#Set-up)
-   [Acknowledgements](#Acknowledgements)

## Motivation

In the United States, the production of solid waste has been on a continuous upward trajectory, reaching an annual average of approximately 292.4 million tons of municipal solid waste (MSW) as of 2020. Notably, the most significant portion of this waste stream comprises paper and paperboard materials.

Despite significant progress made through recycling and composting initiatives, a substantial challenge persists. In 2018, an admirable 94 million tons of MSW were successfully recycled, and an additional 35 million tons were converted into valuable compost for energy generation. Nevertheless, a significant amount of waste, totaling 146 million tons, found its way to landfills, representing a critical area where further sustainable solutions are needed.

Moreover, the emergence of quantum convolutional neural networks presents a groundbreaking opportunity to address these waste management challenges. QCNNs harness the unparalleled computational power of quantum computing, potentially surpassing the capabilities of classical CNNs. By integrating quantum technology into recycling systems, we aim to redefine the efficiency and accuracy of material classification. Qausar is driven by the belief that quantum computing can lead us to innovative breakthroughs in recycling technology, allowing us to reduce waste, conserve resources, and move closer to a more sustainable future.

## Project description

The future of recycling systems is at an exciting crossroads, where cutting-edge technology intersects with environmental sustainability. In this pioneering project, we at Qausar (Quantum Automated System for Advanced Recycling) delve into the realm of quantum convolutional neural networks (QCNNs) to revolutionize recycling efficiency and accuracy, surpassing the capabilities of traditional convolutional neural networks (CNNs). Our research centers on the development of two distinct models: a classical CNN and a quantum CNN, aiming to determine whether QCNNs can pave the way for a more sustainable recycling paradigm.

## How it works

We use the quantum convolution neural network (QCNN) to build the machine learning algorithm. Unlike the convolutional neural network (CNN), which is using classical convolution, quantum convolution utilized the advantage of quantum computing by extending the idea of classical convolution. The appracoh we are using to achieve quantum convolution goes as follows:

we embed a small region of the image into the quantum circuit (2 by 2 square region in our implementation) by parametrized rotations applied to the qubits initialized in the ground state.

Then, We apply a layer of random unitary circuit U on the system to the qubits we got from previous step.

After that, we measure the quantum system, and obtain a list of classical expectation values. We then map each expectation value into a different channel of a single output pixel.

We iterate the same procedure over other regions until all the regions of the image is scanned, and we got an output object which will be structured as a multi-channel image.

We then apply the classical neural network layers to the processed image.

The primary distinction compared to a classical convolution is that a quantum circuit can produce exceedingly intricate kernels, the calculation of which could, in theory, be challenging for classical computers.

## Model Performance

We compared the performance of classical model and quantum model under the same parameters using various metrics.

Our performance comparision strongly supports the adoption of the quantum machine learning model for tasks where speed, accuracy, and model size efficiency are paramount. However, we acknowledge that addressing the loss metric remains a challenge and an area for potential improvement in future developments. These findings underscore the promising potential of quantum machine learning in various real-world applications and motivate further research to harness its strengths while mitigating its limitations.

## Set up

The code, model can be downloaded from (releases)[https://github.com/1mpossible-code/quasar/releases] tab.

Additionally, the PyPi package can be installed to access [Quasar.py](https://github.com/1mpossible-code/quasar/blob/5c6b9ff9319ddaac8e55c204b1a1543f66a62666/app/src/Quasar.py) created for easy access by our team.
The documentation is in progress, so for now, you can use the code as a reference.

```bash
pip install quantum-automated-system-for-advanced-recycling
```

Python:

```python3
from Quasar import Quasar
```

## Acknowledgements

Core Devs: Team "rm -rf /": Maksym Yemelianenko, Qianxi Chen, Ilayda Dilek, Ethan Feldman, Karan Shah, Yufei Zhen

This project was created at the 2023 hAQathon ((AQ = AI + Quantum)) at NYU Tandon: the first quantum computing hackathon hosted at NYU Tandon.

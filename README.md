
<h2 align="center">A napari plugin for cellulus</h2>



- **[Introduction](#introduction)**
- **[Installation](#installation)**
- **[Getting Started](#getting-started)**
- **[Citation](#citation)**
- **[Issues](#issues)**

### Introduction

This repository hosts the code for the napari plugin built around **cellulus**, which was described in the **[preprint](https://arxiv.org/pdf/2310.08501.pdf)** titled **Unsupervised Learning of *Object-Centric Embeddings* for Cell Instance Segmentation in Microscopy Images**.

*cellulus* is a deep learning based method which can be used to obtain instance-segmentation of objects in microscopy images in an unsupervised fashion i.e. requiring no ground truth labels during training.

The main source repository for *cellulus* lives **[here](https://github.com/funkelab/cellulus)**.

### Installation

One could execute these lines of code below to create a new environment and install dependencies.

1. Create a new environment called `napari-cellulus`:

```bash
conda create -y -n napari-cellulus python==3.9
```

2. Activate the newly-created environment:

```
conda activate napari-cellulus
```

3a. If using a GPU, install pytorch cuda dependencies:

```bash
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3b. otherwise (if using a CPU or MPS), run:

```bash
pip install torch torchvision
```

4. Install the package from github:

```bash
pip install git+https://github.com/funkelab/napari-cellulus.git
```

### Getting Started

Run the following commands in a terminal window:
```
conda activate napari-cellulus
napari
```
[demo_cellulus.webm](https://github.com/funkelab/napari-cellulus/assets/34229641/35cb09de-c875-487d-9890-86082dcd95b2)




### Citation

If you find our work useful in your research, please consider citing:


```bibtex
@misc{wolf2023unsupervised,
      title={Unsupervised Learning of Object-Centric Embeddings for Cell Instance Segmentation in Microscopy Images},
      author={Steffen Wolf and Manan Lalit and Henry Westmacott and Katie McDole and Jan Funke},
      year={2023},
      eprint={2310.08501},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### Issues

If you encounter any problems, please **[file an issue](https://github.com/funkelab/napari-cellulus/issues)** along with a description.

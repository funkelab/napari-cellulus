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

To install the latest version:

```
conda create -y -n napari-cellulus python=3.9
conda activate napari-cellulus
pip install torch torchvision
git clone https://github.com/funkelab/napari-cellulus.git
cd napari-cellulus
pip install -e .
```

### Getting Started

Run the following commands in a terminal window
```
conda activate napari-cellulus
napari
```

Next, select `Cellulus` from the `Plugins` drop-down menu.

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

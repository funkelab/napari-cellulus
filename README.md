<p align="center">
  <img src="https://github.com/funkelab/napari-cellulus/assets/34229641/39fd381e-78f3-40da-9a03-1b1f17c22044" width=300 />
</p>
<h2 align="center">A _napari_ plugin for _cellulus_</h2>

- **[Introduction](#introduction)**
- **[Installation](#installation)**
- **[Getting Started](#getting-started)**
- **[Citation](#citation)**
- **[Issues](#issues)**

### Introduction

This repository hosts the code for the _napari_ plugin built around **Cellulus** described in the **[preprint](https://arxiv.org/pdf/2310.08501.pdf)** titled **Unsupervised Learning of *Object-Centric Embeddings* for Cell Instance Segmentation in Microscopy Images**.

Cellulus is a deep learning based method which can be used to obtain instance-segmentation of objects in microscopy images in an unsupervised fashion i.e. requiring no ground truth labels during training.

The main source repository for Cellulus lives **[here](https://github.com/funkelab/cellulus)**.

### Installation

To install the latest development version :

```
pip install git+https://github.com/funkelab/napari-cellulus.git
```

### Getting Started



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

If you encounter any problems, please **[file an issue]** along with a detailed description.

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/funkelab/napari-cellulus/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

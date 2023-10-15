from typing import List

import napari
from magicgui import magic_factory
from napari.qt.threading import FunctionWorker, thread_worker
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
)
from superqt import QCollapsible


class SegmentationWidget(QScrollArea):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # define components
        self.method_description_label = QLabel(
            '<small>Unsupervised Learning of Object-Centric Embeddings<br>for Cell Instance Segmentation in Microscopy Images.<br>If you are using this in your research, please <a href="https://github.com/funkelab/cellulus#citation" style="color:gray;">cite us</a>.</small><br><small><tt><a href="https://github.com/funkelab/cellulus" style="color:gray;">https://github.com/funkelab/cellulus</a></tt></small>'
        )

        # define layout
        outer_layout = QVBoxLayout()

        # inner layout
        grid_0 = QGridLayout()
        grid_0.addWidget(self.method_description_label, 0, 1, 1, 1)
        grid_0.setSpacing(10)

        # Add train configs widget
        collapsible_train_configs = QCollapsible("Train Configs", self)
        collapsible_train_configs.addWidget(self.create_train_configs_widget)

        # Add model configs widget
        collapsible_model_configs = QCollapsible("Model Configs", self)
        collapsible_model_configs.addWidget(self.create_model_configs_widget)

        # Add segment widget
        collapsible_0 = QCollapsible("Inference", self)
        collapsible_0.addWidget(self.segment_widget)

        outer_layout.addLayout(grid_0)
        outer_layout.addWidget(collapsible_train_configs)
        outer_layout.addWidget(collapsible_model_configs)
        outer_layout.addWidget(collapsible_0)
        self.setLayout(outer_layout)
        self.setFixedWidth(400)

    @property
    def create_train_configs_widget(self):
        @magic_factory(
            call_button="Save", device={"choices": ["cuda:0", "cpu", "mps"]}
        )
        def train_configs_widget(
            crop_size: int = 252,
            batch_size: int = 8,
            max_iterations: int = 100_000,
            initial_learning_rate: float = 4e-5,
            temperature: float = 10.0,
            regularizer_weight: float = 1e-5,
            reduce_mean: bool = True,
            density: float = 0.1,
            kappa: float = 10.0,
            save_model_every: int = 1e3,
            save_snapshot_every: int = 1e3,
            num_workers: int = 8,
            device="mps",
        ):
            # Specify what should happen when 'Save' button is pressed
            pass

        if not hasattr(self, "__create_train_configs_widget"):
            self.__create_train_configs_widget = train_configs_widget()
            self.__create_train_configs_widget_native = (
                self.__create_train_configs_widget.native
            )
        return self.__create_train_configs_widget_native

    @property
    def create_model_configs_widget(self):
        @magic_factory(call_button="Save")
        def model_configs_widget(
            num_fmaps: int = 256,
            fmap_inc_factor: int = 3,
            features_in_last_layer: int = 64,
            downsampling_factors: int = 2,
            downsampling_layers: int = 1,
            initialize: bool = True,
        ):
            # Specify what should happen when 'Save' button is pressed
            pass

        if not hasattr(self, "__create_model_configs_widget"):
            self.__create_model_configs_widget = model_configs_widget()
            self.__create_model_configs_widget_native = (
                self.__create_model_configs_widget.native
            )
        return self.__create_model_configs_widget_native

    @property
    def segment_widget(self):
        @magic_factory(call_button="Segment")
        def segment(
            raw: napari.layers.Image,
            crop_size: int = 252,
            p_salt_pepper: float = 0.1,
            num_infer_iterations: int = 16,
            bandwidth: int = 7,
            min_size: int = 25,
        ) -> FunctionWorker[List[napari.types.LayerDataTuple]]:
            @thread_worker(
                connect={"returned": lambda: self.set_buttons("paused")},
                progress={"total": 0, "desc": "Segmenting"},
            )
            def async_segment(
                raw: napari.layers.Image,
                crop_size: int,
                p_salt_pepper: float,
                num_infer_iterations: int,
                bandwidth: int,
                min_size: int,
            ) -> List[napari.types.LayerDataTuple]:

                return async_segment(
                    raw,
                    crop_size=crop_size,
                    p_salt_pepper=p_salt_pepper,
                    num_infer_iterations=num_infer_iterations,
                    bandwidth=bandwidth,
                    min_size=min_size,
                )

        if not hasattr(self, "__segment_widget"):
            self.__segment_widget = segment()
            self.__segment_widget_native = self.__segment_widget.native
        return self.__segment_widget_native

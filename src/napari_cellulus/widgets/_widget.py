from typing import List

import napari
from magicgui import magic_factory
from napari.qt.threading import FunctionWorker, thread_worker
from qtpy.QtWidgets import (
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)
from superqt import QCollapsible


class SegmentationWidget(QScrollArea):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # define components
        logo_path = ""
        self.logo_label = QLabel(f'<h1><img src="{logo_path}">Cellulus</h1>')
        self.method_description_label = QLabel(
            '<small>Unsupervised Learning of Object-Centric Embeddings<br>for Cell Instance Segmentation in Microscopy Images.<br>If you are using this in your research, please <a href="https://github.com/funkelab/cellulus#citation" style="color:gray;">cite us</a>.</small>'
        )

        self.download_data_label = QLabel("<h3>Download Data</h3>")
        self.data_dir_label = QLabel("Data Directory")
        self.data_dir_pushbutton = QPushButton("Browse")
        self.data_dir_pushbutton.setMaximumWidth(280)
        # self.data_dir_pushbutton.clicked.connect(self._prepare_data_dir)

        self.object_size_label = QLabel("Rough Object size [px]")
        self.object_size_edit = QLineEdit("30")

        # define layout
        outer_layout = QVBoxLayout()

        # inner layout
        grid_0 = QGridLayout()
        grid_0.addWidget(self.logo_label, 0, 0, 1, 1)
        grid_0.addWidget(self.method_description_label, 0, 1, 1, 1)
        grid_0.setSpacing(10)

        # Add segment widget
        collapsible_0 = QCollapsible("Inference", self)
        collapsible_0.addWidget(self.segment_widget)

        outer_layout.addLayout(grid_0)
        outer_layout.addWidget(collapsible_0)
        self.setLayout(outer_layout)
        self.setFixedWidth(500)

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

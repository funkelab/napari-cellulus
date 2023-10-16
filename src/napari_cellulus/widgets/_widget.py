import os
from typing import List

import napari
import torch
from cellulus.criterions import get_loss
from cellulus.models import get_model
from cellulus.train import train_iteration
from magicgui import magic_factory

# widget stuff
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from napari.qt.threading import FunctionWorker, thread_worker
from PyQt5.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible

from ..dataset import NapariDataset

# local package imports
from ..gui_helpers import MplCanvas, layer_choice_widget


class SegmentationWidget(QScrollArea):
    def __init__(self, napari_viewer):
        super().__init__()
        self.widget = QWidget()
        self.viewer = napari_viewer

        # define components
        method_description_label = QLabel(
            '<small>Unsupervised Learning of Object-Centric Embeddings<br>for Cell Instance Segmentation in Microscopy Images.<br>If you are using this in your research, please <a href="https://github.com/funkelab/cellulus#citation" style="color:gray;">cite us</a>.</small><br><small><tt><a href="https://github.com/funkelab/cellulus" style="color:gray;">https://github.com/funkelab/cellulus</a></tt></small>'
        )

        # define layout
        outer_layout = QVBoxLayout()

        # Initialize train configs widget
        collapsible_train_configs = QCollapsible("Train Configs", self)
        collapsible_train_configs.addWidget(self.create_train_configs_widget)

        # Initialize model configs widget
        collapsible_model_configs = QCollapsible("Model Configs", self)
        collapsible_model_configs.addWidget(self.create_model_configs_widget)

        # Initialize loss/iterations widget
        self.progress_plot = MplCanvas(self, width=5, height=3, dpi=100)
        toolbar = NavigationToolbar(self.progress_plot, self)
        progress_plot_layout = QVBoxLayout()
        progress_plot_layout.addWidget(toolbar)
        progress_plot_layout.addWidget(self.progress_plot)
        self.loss_plot = None
        self.val_plot = None
        plot_container_widget = QWidget()
        plot_container_widget.setLayout(progress_plot_layout)

        # Initialize Layer Choice
        self.raw_selector = layer_choice_widget(
            self.viewer, annotation=napari.layers.Image, name="raw"
        )

        # Initialize Checkboxes
        self.s_checkbox = QCheckBox("s")
        self.c_checkbox = QCheckBox("c")
        self.t_checkbox = QCheckBox("t")
        self.z_checkbox = QCheckBox("z")
        self.y_checkbox = QCheckBox("y")
        self.x_checkbox = QCheckBox("x")

        axis_layout = QHBoxLayout()
        axis_layout.addWidget(self.s_checkbox)
        axis_layout.addWidget(self.c_checkbox)
        axis_layout.addWidget(self.t_checkbox)
        axis_layout.addWidget(self.z_checkbox)
        axis_layout.addWidget(self.y_checkbox)
        axis_layout.addWidget(self.x_checkbox)
        axis_selector = QGroupBox("Axis Names:")
        axis_selector.setLayout(axis_layout)

        # Initialize Train Button
        train_button = QPushButton("Train!", self)
        train_button.clicked.connect(self.prepare_for_training)

        # Add everything to outer_layout
        outer_layout.addWidget(method_description_label)
        outer_layout.addWidget(collapsible_train_configs)
        outer_layout.addWidget(collapsible_model_configs)
        outer_layout.addWidget(plot_container_widget)
        outer_layout.addWidget(self.raw_selector.native)
        outer_layout.addWidget(axis_selector)
        outer_layout.addWidget(train_button)

        outer_layout.setSpacing(20)
        self.widget.setLayout(outer_layout)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setWidget(self.widget)
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
            num_workers: int = 8,
            control_point_spacing: int = 64,
            control_point_jitter: float = 2.0,
            device="mps",
        ):
            # Specify what should happen when 'Save' button is pressed
            self.train_config = {
                "crop_size": crop_size,
                "batch_size": batch_size,
                "max_iterations": max_iterations,
                "initial_learning_rate": initial_learning_rate,
                "temperature": temperature,
                "regularizer_weight": regularizer_weight,
                "reduce_mean": reduce_mean,
                "density": density,
                "kappa": kappa,
                "num_workers": num_workers,
                "control_point_spacing": control_point_spacing,
                "control_point_jitter": control_point_jitter,
                "device": device,
            }

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
            self.model_config = {
                "num_fmaps": num_fmaps,
                "fmap_inc_factor": fmap_inc_factor,
                "features_in_last_layer": features_in_last_layer,
                "downsampling_factors": downsampling_factors,
                "downsampling_layers": downsampling_layers,
                "initialize": initialize,
            }

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

    def get_selected_axes(self):
        names = []
        for name, checkbox in zip(
            "sctzyx",
            [
                self.s_checkbox,
                self.c_checkbox,
                self.t_checkbox,
                self.z_checkbox,
                self.y_checkbox,
                self.x_checkbox,
            ],
        ):
            if checkbox.isChecked():
                names.append(name)
        return names

    def prepare_for_training(self):

        # check if train_config object exists
        # check if model_config object exists

        if self.train_config is None:
            pass  # TODO
        if self.model_config is None:
            pass  # TODO

        self.__training_generator = self.train_napari(iteration=0)  # TODO
        self.__training_generator.yielded.connect(self.on_yield)
        self.__training_generator.returned.connect(self.on_return)
        self.__training_generator.start()

    @thread_worker
    def train_napari(self, iteration=0):

        # Turn layer into dataset
        train_dataset = NapariDataset(
            layer=self.raw_selector.value,
            axis_names=self.get_selected_axes(),
            crop_size=self.train_config["crop_size"],
            control_point_spacing=self.train_config["control_point_spacing"],
            control_point_jitter=self.train_config["control_point_jitter"],
        )

        if not os.path.exists("models"):
            os.makedirs("models")

        # create train dataloader
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.train_config["batch_size"],
            drop_last=True,
            num_workers=self.train_config["num_workers"],
            pin_memory=True,
        )

        downsampling_factors = [
            [
                self.model_config["downsampling_factors"],
            ]
            * train_dataset.get_num_spatial_dims()
        ] * self.model_config["downsampling_layers"]

        # set model
        model = get_model(
            in_channels=train_dataset.get_num_channels(),
            out_channels=train_dataset.get_num_spatial_dims(),
            num_fmaps=self.model_config["num_fmaps"],
            fmap_inc_factor=self.model_config["fmap_inc_factor"],
            features_in_last_layer=self.model_config["features_in_last_layer"],
            downsampling_factors=[
                tuple(factor) for factor in downsampling_factors
            ],
            num_spatial_dims=train_dataset.get_num_spatial_dims(),
        )

        # set device
        device = torch.device(self.train_config["device"])

        model = model.to(device)

        # initialize model weights
        if self.model_config["initialize"]:
            for _name, layer in model.named_modules():
                if isinstance(layer, torch.nn.modules.conv._ConvNd):
                    torch.nn.init.kaiming_normal_(
                        layer.weight, nonlinearity="relu"
                    )

        # set loss
        criterion = get_loss(
            regularizer_weight=self.train_config["regularizer_weight"],
            temperature=self.train_config["temperature"],
            kappa=self.train_config["kappa"],
            density=self.train_config["density"],
            num_spatial_dims=train_dataset.get_num_spatial_dims(),
            reduce_mean=self.train_config["reduce_mean"],
            device=device,
        )

        # set optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.train_config["initial_learning_rate"],
        )

        # set scheduler:

        def lambda_(iteration):
            return pow(
                (1 - ((iteration) / self.train_config["max_iterations"])), 0.9
            )

        # resume training
        start_iteration = 0

        # TODO
        # if self.model_config.checkpoint is None:
        #    pass
        # else:
        #    print(f"Resuming model from {self.model_config.checkpoint}")
        #    state = torch.load(self.model_config.checkpoint, map_location=device)
        #    start_iteration = state["iteration"] + 1
        #    lowest_loss = state["lowest_loss"]
        #    model.load_state_dict(state["model_state_dict"], strict=True)
        #    optimizer.load_state_dict(state["optim_state_dict"])
        #    logger.data = state["logger_data"]

        # call `train_iteration`
        for iteration, batch in zip(
            range(start_iteration, self.train_config["max_iterations"]),
            train_dataloader,
        ):
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda_, last_epoch=iteration - 1
            )

            train_loss, prediction = train_iteration(
                batch,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            scheduler.step()
            yield (iteration, train_loss)

    def on_yield(self, step_data):
        # TODO
        iteration, loss = step_data
        print(iteration, loss)

    def on_return(self):
        pass  # TODO

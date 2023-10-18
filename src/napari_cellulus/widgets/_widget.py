import os

import gunpowder as gp
import napari
import numpy as np
import torch
from cellulus.criterions import get_loss
from cellulus.models import get_model
from cellulus.train import train_iteration
from cellulus.utils.mean_shift import mean_shift_segmentation
from magicgui import magic_factory

# widget stuff
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import distance_transform_edt as dtedt
from superqt import QCollapsible

from ..dataset import NapariDataset
from ..gp.nodes.napari_image_source import NapariImageSource

# local package imports
from ..gui_helpers import MplCanvas, layer_choice_widget


class SegmentationWidget(QScrollArea):
    def __init__(self, napari_viewer):
        super().__init__()
        self.widget = QWidget()
        self.viewer = napari_viewer

        # initialize train_config and model_config

        self.train_config = None
        self.model_config = None
        self.segment_config = None

        # initialize losses and iterations
        self.losses = []
        self.iterations = []

        # initialize mode. this will change to 'training' and 'segmentring' later
        self.mode = "configuring"

        # initialize UI components
        method_description_label = QLabel(
            '<small>Unsupervised Learning of Object-Centric Embeddings<br>for Cell Instance Segmentation in Microscopy Images.<br>If you are using this in your research, please <a href="https://github.com/funkelab/cellulus#citation" style="color:gray;">cite us</a>.</small><br><small><tt><a href="https://github.com/funkelab/cellulus" style="color:gray;">https://github.com/funkelab/cellulus</a></tt></small>'
        )

        # specify layout
        outer_layout = QVBoxLayout()

        # Initialize object size widget
        object_size_label = QLabel(self)
        object_size_label.setText("Object Size [px]:")
        self.object_size_line = QLineEdit(self)
        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(object_size_label)
        hbox_layout.addWidget(self.object_size_line)
        object_size_box = QGroupBox("")
        object_size_box.setLayout(hbox_layout)

        # Initialize train configs widget
        collapsible_train_configs = QCollapsible("Train Configs", self)
        collapsible_train_configs.addWidget(self.create_train_configs_widget)

        # Initialize model configs widget
        collapsible_model_configs = QCollapsible("Model Configs", self)
        collapsible_model_configs.addWidget(self.create_model_configs_widget)

        # Initialize loss/iterations widget
        self.canvas = MplCanvas(self, width=5, height=3, dpi=100)
        toolbar = NavigationToolbar(self.canvas, self)
        canvas_layout = QVBoxLayout()
        canvas_layout.addWidget(toolbar)
        canvas_layout.addWidget(self.canvas)
        if len(self.iterations) == 0:
            self.loss_plot = self.canvas.axes.plot(
                [], [], label="Training Loss"
            )[0]
            self.canvas.axes.legend()
            self.canvas.axes.set_xlabel("Iterations")
            self.canvas.axes.set_ylabel("Loss")
        plot_container_widget = QWidget()
        plot_container_widget.setLayout(canvas_layout)

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
        self.train_button = QPushButton("Train", self)
        self.train_button.clicked.connect(self.prepare_for_training)

        # Initialize Model Configs widget
        collapsible_segment_configs = QCollapsible("Inference Configs", self)
        collapsible_segment_configs.addWidget(
            self.create_segment_configs_widget
        )

        # Initialize Segment Button
        self.segment_button = QPushButton("Segment", self)
        self.segment_button.clicked.connect(self.prepare_for_segmenting)

        # Initialize progress bar
        self.pbar = QProgressBar(self)

        # Add all components to outer_layout

        outer_layout.addWidget(method_description_label)
        outer_layout.addWidget(object_size_box)
        outer_layout.addWidget(collapsible_train_configs)
        outer_layout.addWidget(collapsible_model_configs)
        outer_layout.addWidget(plot_container_widget)
        outer_layout.addWidget(self.raw_selector.native)
        outer_layout.addWidget(axis_selector)
        outer_layout.addWidget(self.train_button)
        outer_layout.addWidget(collapsible_segment_configs)
        outer_layout.addWidget(self.segment_button)
        outer_layout.addWidget(self.pbar)

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
    def create_segment_configs_widget(self):
        @magic_factory(call_button="Save")
        def segment_configs_widget(
            crop_size: int = 252,
            p_salt_pepper: float = 0.1,
            num_infer_iterations: int = 16,
            bandwidth: int = 7,
            reduction_probability: float = 0.1,
            min_size: int = 25,
            grow_distance: int = 3,
            shrink_distance: int = 6,
        ):
            # Specify what should happen when 'Save' button is pressed
            self.segment_config = {
                "crop_size": crop_size,
                "p_salt_pepper": p_salt_pepper,
                "num_infer_iterations": num_infer_iterations,
                "bandwidth": bandwidth,
                "reduction_probability": reduction_probability,
                "min_size": min_size,
                "grow_distance": grow_distance,
                "shrink_distance": shrink_distance,
            }

        if not hasattr(self, "__create_segment_configs_widget"):
            self.__create_segment_configs_widget = segment_configs_widget()
            self.__create_segment_configs_widget_native = (
                self.__create_segment_configs_widget.native
            )
        return self.__create_segment_configs_widget_native

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
        if self.train_config is None:
            # set default values
            self.train_config = {
                "crop_size": 252,
                "batch_size": 8,
                "max_iterations": 100_000,
                "initial_learning_rate": 4e-5,
                "temperature": 10.0,
                "regularizer_weight": 1e-5,
                "reduce_mean": True,
                "density": 0.1,
                "kappa": 10.0,
                "num_workers": 8,
                "control_point_spacing": 64,
                "control_point_jitter": 2.0,
                "device": "mps",
            }

        # check if model_config object exists
        if self.model_config is None:
            self.model_config = {
                "num_fmaps": 256,
                "fmap_inc_factor": 3,
                "features_in_last_layer": 64,
                "downsampling_factors": 2,
                "downsampling_layers": 1,
                "initialize": True,
            }
        self.update_mode(self.sender())

        if self.mode == "training":
            self.worker = self.train_napari()
            self.worker.yielded.connect(self.on_yield)
            self.worker.returned.connect(self.on_return)
            self.worker.start()
        elif self.mode == "configuring":
            state = {
                "iteration": self.iterations[-1],
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "iterations": self.iterations,
                "losses": self.losses,
            }
            filename = os.path.join("models", "last.pth")
            torch.save(state, filename)
            self.worker.quit()

    def update_mode(self, sender):

        if self.train_button.text() == "Train" and sender == self.train_button:
            self.train_button.setText("Pause")
            self.mode = "training"
        elif (
            self.train_button.text() == "Pause" and sender == self.train_button
        ):
            self.train_button.setText("Train")
            self.mode = "configuring"
        elif (
            self.segment_button.text() == "Segment"
            and sender == self.segment_button
        ):
            self.segment_button.setText("Pause")
            self.mode = "segmenting"
        elif (
            self.segment_button.text() == "Pause"
            and sender == self.segment_button
        ):
            self.segment_button.setText("Segment")
            self.mode = "configuring"

    @thread_worker
    def train_napari(self):

        # Turn layer into dataset
        self.dataset = NapariDataset(
            layer=self.raw_selector.value,
            axis_names=self.get_selected_axes(),
            crop_size=self.train_config["crop_size"],
            control_point_spacing=self.train_config["control_point_spacing"],
            control_point_jitter=self.train_config["control_point_jitter"],
        )

        if not os.path.exists("models"):
            os.makedirs("models")

        # create train dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.train_config["batch_size"],
            drop_last=True,
            num_workers=self.train_config["num_workers"],
            pin_memory=True,
        )

        downsampling_factors = [
            [
                self.model_config["downsampling_factors"],
            ]
            * self.dataset.get_num_spatial_dims()
        ] * self.model_config["downsampling_layers"]

        # set model
        self.model = get_model(
            in_channels=self.dataset.get_num_channels(),
            out_channels=self.dataset.get_num_spatial_dims(),
            num_fmaps=self.model_config["num_fmaps"],
            fmap_inc_factor=self.model_config["fmap_inc_factor"],
            features_in_last_layer=self.model_config["features_in_last_layer"],
            downsampling_factors=[
                tuple(factor) for factor in downsampling_factors
            ],
            num_spatial_dims=self.dataset.get_num_spatial_dims(),
        )

        # set device
        device = torch.device(self.train_config["device"])

        self.model = self.model.to(device)

        # initialize model weights
        if self.model_config["initialize"]:
            for _name, layer in self.model.named_modules():
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
            num_spatial_dims=self.dataset.get_num_spatial_dims(),
            reduce_mean=self.train_config["reduce_mean"],
            device=device,
        )

        # set optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config["initial_learning_rate"],
        )

        # set scheduler:

        def lambda_(iteration):
            return pow(
                (1 - ((iteration) / self.train_config["max_iterations"])), 0.9
            )

        # resume training
        if len(self.iterations) == 0:
            start_iteration = 0
        else:
            start_iteration = self.iterations[-1]

        if not os.path.exists("models/last.pth"):
            pass
        else:
            print("Resuming model from 'models/last.pth'")
            state = torch.load("models/last.pth", map_location=device)
            start_iteration = state["iteration"] + 1
            self.iterations = state["iterations"]
            self.losses = state["losses"]
            self.model.load_state_dict(state["model_state_dict"], strict=True)
            self.optimizer.load_state_dict(state["optim_state_dict"])

        # call `train_iteration`
        for iteration, batch in zip(
            range(start_iteration, self.train_config["max_iterations"]),
            self.dataloader,
        ):
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda_, last_epoch=iteration - 1
            )

            train_loss, prediction = train_iteration(
                batch,
                model=self.model,
                criterion=criterion,
                optimizer=self.optimizer,
                device=device,
            )
            scheduler.step()
            yield (iteration, train_loss)

    def on_yield(self, step_data):
        if self.mode == "training":
            iteration, loss = step_data
            print(iteration, loss)
            self.iterations.append(iteration)
            self.losses.append(loss)
            self.update_canvas()
        elif self.mode == "segmenting":
            print(step_data)
            self.pbar.setValue(step_data)

    def update_canvas(self):
        self.loss_plot.set_xdata(self.iterations)
        self.loss_plot.set_ydata(self.losses)
        self.canvas.axes.relim()
        self.canvas.axes.autoscale_view()
        self.canvas.draw()

    def on_return(self):
        pass  # TODO

    def prepare_for_segmenting(self):
        # check if segment_config exists
        if self.segment_config is None:
            self.segment_config = {
                "crop_size": 252,
                "p_salt_pepper": 0.1,
                "num_infer_iterations": 16,
                "bandwidth": None,
                "reduction_probability": 0.1,
                "min_size": None,
                "grow_distance": 3,
                "shrink_distance": 6,
            }

        # update mode
        self.update_mode(self.sender())

        if self.mode == "segmenting":
            self.worker = self.segment_napari()
            self.worker.yielded.connect(self.on_yield)
            self.worker.returned.connect(self.on_return)
            self.worker.start()
        elif self.mode == "configuring":
            self.worker.quit()

    @thread_worker
    def segment_napari(self):
        raw = self.raw_selector.value

        if self.segment_config["bandwidth"] is None:
            self.segment_config["bandwidth"] = int(
                0.5 * float(self.object_size_line.text())
            )
        if self.segment_config["min_size"] is None:
            self.segment_config["min_size"] = int(
                0.1 * np.pi * (float(self.object_size_line.text()) ** 2) / 4
            )
        self.model.eval()

        num_spatial_dims = self.dataset.num_spatial_dims
        num_channels = self.dataset.num_channels
        spatial_dims = self.dataset.spatial_dims
        num_samples = self.dataset.num_samples

        print(
            f"Num spatial dims {num_spatial_dims} num channels {num_channels} spatial_dims {spatial_dims} num_samples {num_samples}"
        )

        crop_size = (self.segment_config["crop_size"],) * num_spatial_dims
        device = self.train_config["device"]

        if num_channels == 0:
            num_channels = 1

        voxel_size = gp.Coordinate((1,) * num_spatial_dims)
        self.model.set_infer(
            p_salt_pepper=self.segment_config["p_salt_pepper"],
            num_infer_iterations=self.segment_config["num_infer_iterations"],
            device=device,
        )

        print(f"Current device is {device}")

        input_shape = gp.Coordinate((1, num_channels, *crop_size))
        output_shape = gp.Coordinate(
            self.model(
                torch.zeros(
                    (1, num_channels, *crop_size), dtype=torch.float32
                ).to(device)
            ).shape
        )
        input_size = (
            gp.Coordinate(input_shape[-num_spatial_dims:]) * voxel_size
        )
        output_size = (
            gp.Coordinate(output_shape[-num_spatial_dims:]) * voxel_size
        )
        context = (input_size - output_size) / 2

        raw_key = gp.ArrayKey("RAW")
        prediction_key = gp.ArrayKey("PREDICT")
        scan_request = gp.BatchRequest()
        scan_request.add(raw_key, input_size)
        scan_request.add(prediction_key, output_size)

        predict = gp.torch.Predict(
            self.model,
            inputs={"raw": raw_key},
            outputs={0: prediction_key},
            array_specs={prediction_key: gp.ArraySpec(voxel_size=voxel_size)},
        )
        pipeline = NapariImageSource(
            raw,
            raw_key,
            gp.ArraySpec(
                gp.Roi(
                    (0,) * num_spatial_dims,
                    raw.data.shape[-num_spatial_dims:],
                ),
                voxel_size=voxel_size,
            ),
            spatial_dims,
        )

        if num_samples == 0 and num_channels == 0:
            pipeline += (
                gp.Pad(raw_key, context)
                + gp.Unsqueeze([raw_key], 0)
                + gp.Unsqueeze([raw_key], 0)
                + predict
                + gp.Scan(scan_request)
            )
        elif num_samples != 0 and num_channels == 0:
            pipeline += (
                gp.Pad(raw_key, context)
                + gp.Unsqueeze([raw_key], 1)
                + predict
                + gp.Scan(scan_request)
            )
        elif num_samples == 0 and num_channels != 0:
            pipeline += (
                gp.Pad(raw_key, context)
                + gp.Unsqueeze([raw_key], 0)
                + predict
                + gp.Scan(scan_request)
            )
        elif num_samples != 0 and num_channels != 0:
            pipeline += (
                gp.Pad(raw_key, context) + predict + gp.Scan(scan_request)
            )

        # request to pipeline for ROI of whole image/volume
        request = gp.BatchRequest()
        request.add(prediction_key, raw.data.shape[-num_spatial_dims:])
        counter = 0
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)
            yield counter
            counter += 0.1

        prediction = batch.arrays[prediction_key].data
        colormaps = ["red", "green", "blue"]
        prediction_layers = [
            (
                prediction[:, i : i + 1, ...].copy(),
                {
                    "name": "offset-" + "zyx"[num_spatial_dims - i]
                    if i < num_spatial_dims
                    else "std",
                    "colormap": colormaps[num_spatial_dims - i]
                    if i < num_spatial_dims
                    else "gray",
                    "blending": "additive",
                },
                "image",
            )
            for i in range(num_spatial_dims + 1)
        ]

        labels = np.zeros_like(prediction[:, 0:1, ...].data, dtype=np.uint64)
        for sample in range(num_samples):
            embeddings = prediction[sample]
            embeddings_std = embeddings[-1, ...]
            embeddings_mean = embeddings[np.newaxis, :num_spatial_dims, ...]
            segmentation = mean_shift_segmentation(
                embeddings_mean,
                embeddings_std,
                self.segment_config["bandwidth"],
                self.segment_config["min_size"],
                self.segment_config["reduction_probability"],
            )
            labels[sample, 0, ...] = segmentation

        pp_labels = np.zeros_like(
            prediction[:, 0:1, ...].data, dtype=np.uint64
        )
        for sample in range(num_samples):
            segmentation = labels[sample, 0]
            distance_foreground = dtedt(segmentation == 0)
            expanded_mask = (
                distance_foreground < self.inference_config["grow_distance"]
            )
            distance_background = dtedt(expanded_mask)
            segmentation[
                distance_background < self.inference_config["shrink_distance"]
            ] = 0
            pp_labels[sample, 0, ...] = segmentation
        return (
            prediction_layers
            + [(labels, {"name": "Segmentation"}, "labels")]
            + [(pp_labels, {"name": "Post Processed"}, "labels")]
        )

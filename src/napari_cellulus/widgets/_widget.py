"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
import contextlib
import dataclasses

# python built in libraries
from pathlib import Path
from typing import List, Optional

# github repo libraries
import gunpowder as gp

# pip installed libraries
import napari
import numpy as np
import torch
from cellulus.configs.model_config import ModelConfig
from cellulus.configs.train_config import TrainConfig
from cellulus.criterions import get_loss
from cellulus.models import get_model
from cellulus.utils.mean_shift import mean_shift_segmentation
from magicgui import magic_factory
from magicgui.widgets import Container

# widget stuff
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from napari.qt.threading import FunctionWorker, thread_worker
from qtpy.QtCore import QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible
from tqdm import tqdm

from ..dataset import NapariDataset
from ..gp.nodes.napari_image_source import NapariImageSource

# local package imports
from ..gui_helpers import MplCanvas, layer_choice_widget
from ..meta_data import NapariDatasetMetaData


@dataclasses.dataclass
class TrainingStats:
    iteration: int = 0
    losses: list[float] = dataclasses.field(default_factory=list)
    iterations: list[int] = dataclasses.field(default_factory=list)

    def reset(self):
        self.iteration = 0
        self.losses = []
        self.iterations = []

    def load(self, other):
        self.iteration = other.iteration
        self.losses = other.losses
        self.iterations = other.iterations


################################## GLOBALS ####################################
_train_config: Optional[TrainConfig] = None
_model_config: Optional[ModelConfig] = None
_model: Optional[torch.nn.Module] = None
_optimizer: Optional[torch.optim.Optimizer] = None
_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
_training_stats: TrainingStats = TrainingStats()


def get_training_stats():
    global _training_stats
    return _training_stats


def get_train_config(**kwargs):
    global _train_config
    # set dataset configs to None
    kwargs["train_data_config"] = None
    kwargs["validate_data_config"] = None
    if _train_config is None:
        _train_config = TrainConfig(**kwargs)
    elif len(kwargs) > 0:
        for k, v in kwargs.items():
            _train_config.__setattr__(k, v)
    return _train_config


@magic_factory(
    call_button="Save", device={"choices": ["cpu", "cuda:0", "mps"]}
)
def train_config_widget(
    crop_size: list[int] = [252, 252],
    batch_size: int = 8,
    max_iterations: int = 100_000,
    initial_learning_rate: float = 4e-5,
    density: float = 0.1,
    kappa: float = 10.0,
    temperature: float = 10.0,
    regularizer_weight: float = 1e-5,
    reduce_mean: bool = True,
    save_model_every: int = 1_000,
    save_snapshot_every: int = 1_000,
    num_workers: int = 8,
    control_point_spacing: int = 64,
    control_point_jitter: float = 2.0,
    device="cpu",
):
    get_train_config(
        crop_size=crop_size,
        batch_size=batch_size,
        max_iterations=max_iterations,
        initial_learning_rate=initial_learning_rate,
        density=density,
        kappa=kappa,
        temperature=temperature,
        regularizer_weight=regularizer_weight,
        reduce_mean=reduce_mean,
        save_model_every=save_model_every,
        save_snapshot_every=save_snapshot_every,
        num_workers=num_workers,
        control_point_spacing=control_point_spacing,
        control_point_jitter=control_point_jitter,
        device=device,
    )


def get_model_config(**kwargs):
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig(**kwargs)
    elif len(kwargs) > 0:
        for k, v in kwargs.items():
            _model_config.__setattr__(k, v)
    return _model_config


@magic_factory
def model_config_widget(
    num_fmaps: int = 256,
    fmap_inc_factor: int = 3,
    features_in_last_layer: int = 64,
    downsampling_factors: list[list[int]] = [[2, 2]],
):
    get_model_config(
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        features_in_last_layer=features_in_last_layer,
        downsampling_factors=downsampling_factors,
    )


def get_training_state(dataset: Optional[NapariDataset] = None):
    global _model
    global _optimizer
    global _scheduler
    global _model_config
    global _train_config

    # set device
    device = torch.device(_train_config.device)

    if _model_config is None:
        # TODO: deal with hard coded defaults
        _model_config = ModelConfig(num_fmaps=24, fmap_inc_factor=3)
    if _train_config is None:
        _train_config = get_train_config()
    if _model is None:
        # Build model
        _model = get_model(
            in_channels=dataset.get_num_channels(),
            out_channels=dataset.get_num_spatial_dims(),
            num_fmaps=_model_config.num_fmaps,
            fmap_inc_factor=_model_config.fmap_inc_factor,
            features_in_last_layer=_model_config.features_in_last_layer,
            downsampling_factors=[
                tuple(factor) for factor in _model_config.downsampling_factors
            ],
            num_spatial_dims=dataset.get_num_spatial_dims(),
        ).to(device)

        # Weight initialization
        # TODO: move weight initialization to funlib.learn.torch
        for _name, layer in _model.named_modules():
            if isinstance(layer, torch.nn.modules.conv._ConvNd):
                torch.nn.init.kaiming_normal_(
                    layer.weight, nonlinearity="relu"
                )

        _optimizer = torch.optim.Adam(
            _model.parameters(),
            lr=_train_config.initial_learning_rate,
        )

        def lambda_(iteration):
            return pow((1 - ((iteration) / _train_config.max_iterations)), 0.9)

        _scheduler = torch.optim.lr_scheduler.LambdaLR(
            _optimizer, lr_lambda=lambda_
        )
    return (_model, _optimizer, _scheduler)


class TrainWidget(QWidget):
    def __init__(self, napari_viewer):
        # basic initialization
        self.viewer = napari_viewer
        super().__init__()

        # initialize state variables
        self.__training_generator = None

        # Widget layout
        layout = QVBoxLayout()

        # add loss/iterations widget
        self.progress_plot = MplCanvas(self, width=5, height=3, dpi=100)
        toolbar = NavigationToolbar(self.progress_plot, self)
        progress_plot_layout = QVBoxLayout()
        progress_plot_layout.addWidget(toolbar)
        progress_plot_layout.addWidget(self.progress_plot)
        self.loss_plot = None
        self.val_plot = None
        plot_container_widget = QWidget()
        plot_container_widget.setLayout(progress_plot_layout)
        layout.addWidget(plot_container_widget)

        # add raw layer choice
        self.raw_selector = layer_choice_widget(
            self.viewer,
            annotation=napari.layers.Image,
            name="raw",
        )
        layout.addWidget(self.raw_selector.native)

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

        self.axis_selector = QGroupBox("Axis Names:")
        self.axis_selector.setLayout(axis_layout)
        layout.addWidget(self.axis_selector)

        # add buttons
        self.train_button = QPushButton("Train!", self)
        self.train_button.clicked.connect(self.train)
        layout.addWidget(self.train_button)

        # add save and load widgets
        collapsable_save_load_widget = QCollapsible("Save/Load", self)
        collapsable_save_load_widget.addWidget(self.save_widget.native)
        collapsable_save_load_widget.addWidget(self.load_widget.native)

        layout.addWidget(collapsable_save_load_widget)

        # Add segment widget
        collapsable_segment_widget = QCollapsible("Segment", self)
        collapsable_segment_widget.addWidget(self.segment_widget)
        layout.addWidget(collapsable_segment_widget)

        # add feedback button
        self.feedback_button = QPushButton("Feedback!", self)
        self.feedback_button.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl(
                    "https://github.com/funkelab/napari-cellulus/issues/new/choose"
                )
            )
        )
        layout.addWidget(self.feedback_button)

        # activate layout
        self.setLayout(layout)

        # Widget state
        self.model = None
        self.reset_training_state()

        # TODO: Can we do this better?
        # connect napari events
        self.viewer.layers.events.inserted.connect(
            self.__segment_widget.raw.reset_choices
        )
        self.viewer.layers.events.removed.connect(
            self.__segment_widget.raw.reset_choices
        )

        # handle button activations and deactivations
        # buttons: save, load, (train/pause), segment
        self.save_button = self.__save_widget.call_button.native
        self.load_button = self.__load_widget.call_button.native
        self.segment_button = self.__segment_widget.call_button.native
        self.segment_button.clicked.connect(
            lambda: self.set_buttons("segmenting")
        )

        self.set_buttons("initial")

    def set_buttons(self, state: str):
        if state == "training":
            self.train_button.setText("Pause!")
            self.train_button.setEnabled(True)
            self.save_button.setText("Stop training to save!")
            self.save_button.setEnabled(False)
            self.load_button.setText("Stop training to load!")
            self.load_button.setEnabled(False)
            self.segment_button.setText("Stop training to segment!")
            self.segment_button.setEnabled(False)
        if state == "paused":
            self.train_button.setText("Train!")
            self.train_button.setEnabled(True)
            self.save_button.setText("Save")
            self.save_button.setEnabled(True)
            self.load_button.setText("Load")
            self.load_button.setEnabled(True)
            self.segment_button.setText("Segment")
            self.segment_button.setEnabled(True)
        if state == "segmenting":
            self.train_button.setText("Can't train while segmenting!")
            self.train_button.setEnabled(False)
            self.save_button.setText("Can't save while segmenting!")
            self.save_button.setEnabled(False)
            self.load_button.setText("Can't load while segmenting!")
            self.load_button.setEnabled(False)
            self.segment_button.setText("Segmenting...")
            self.segment_button.setEnabled(False)
        if state == "initial":
            self.train_button.setText("Train!")
            self.train_button.setEnabled(True)
            self.save_button.setText("No state to Save!")
            self.save_button.setEnabled(False)
            self.load_button.setText(
                "Load data and test data before loading an old model!"
            )
            self.load_button.setEnabled(False)
            self.segment_button.setText("Segment")
            self.segment_button.setEnabled(True)

    @property
    def segment_widget(self):
        @magic_factory(call_button="Segment")
        def segment(
            raw: napari.layers.Image,
            crop_size: list[int] = [252, 252],
            p_salt_pepper: float = 0.1,
            num_infer_iterations: int = 16,
            bandwidth: int = 7,
            min_size: int = 10,
        ) -> FunctionWorker[List[napari.types.LayerDataTuple]]:
            # TODO: do this better?
            @thread_worker(
                connect={"returned": lambda: self.set_buttons("paused")},
                progress={"total": 0, "desc": "Segmenting"},
            )
            def async_segment(
                raw: napari.layers.Image,
                crop_size: list[int],
                p_salt_pepper: float,
                num_infer_iterations: int,
                bandwidth: int,
                min_size: int,
            ) -> List[napari.types.LayerDataTuple]:

                raw.data = raw.data.astype(np.float32)
                global _model

                assert (
                    _model is not None
                ), "You must train a model before running inference"
                model = _model

                # set in eval mode
                model.eval()

                # device
                device = torch.device(_train_config.device)

                axis_names = self.get_selected_axes()
                meta_data = NapariDatasetMetaData(raw.data.shape, axis_names)

                num_spatial_dims = meta_data.num_spatial_dims
                num_channels = meta_data.num_channels

                if meta_data.num_channels == 0:
                    num_channels = 1

                voxel_size = gp.Coordinate((1,) * num_spatial_dims)
                model.set_infer(
                    p_salt_pepper=p_salt_pepper,
                    num_infer_iterations=num_infer_iterations,
                    device=device,
                )

                # prediction crop size is the size of the scanned tiles to be provided to the model
                input_shape = gp.Coordinate((1, num_channels, *crop_size))

                output_shape = gp.Coordinate(
                    model(
                        torch.zeros(
                            (
                                1,
                                num_channels,
                                *crop_size,
                            ),
                            dtype=torch.float32,
                        ).to(device)
                    ).shape
                )
                input_size = (
                    gp.Coordinate(input_shape[-num_spatial_dims:]) * voxel_size
                )
                output_size = (
                    gp.Coordinate(output_shape[-num_spatial_dims:])
                    * voxel_size
                )

                context = (input_size - output_size) / 2

                raw_key = gp.ArrayKey("RAW")
                prediction_key = gp.ArrayKey("PREDICT")

                scan_request = gp.BatchRequest()
                scan_request.add(raw_key, input_size)
                scan_request.add(prediction_key, output_size)

                predict = gp.torch.Predict(
                    model,
                    inputs={"raw": raw_key},
                    outputs={0: prediction_key},
                    array_specs={
                        prediction_key: gp.ArraySpec(voxel_size=voxel_size)
                    },
                )
                if meta_data.num_samples == 0 and meta_data.num_channels == 0:
                    pipeline = (
                        NapariImageSource(
                            raw,
                            raw_key,
                            gp.ArraySpec(
                                gp.Roi(
                                    (0,) * num_spatial_dims,
                                    raw.data.shape[-num_spatial_dims:],
                                ),
                                voxel_size=voxel_size,
                            ),
                        )
                        + gp.Pad(raw_key, context)
                        + gp.Unsqueeze([raw_key], 0)
                        + gp.Unsqueeze([raw_key], 0)
                        + predict
                        + gp.Scan(scan_request)
                    )
                elif (
                    meta_data.num_samples != 0 and meta_data.num_channels == 0
                ):
                    pipeline = (
                        NapariImageSource(
                            raw,
                            raw_key,
                            gp.ArraySpec(
                                gp.Roi(
                                    (0,) * num_spatial_dims,
                                    raw.data.shape[-num_spatial_dims:],
                                ),
                                voxel_size=voxel_size,
                            ),
                        )
                        + gp.Pad(raw_key, context)
                        + gp.Unsqueeze([raw_key], 1)
                        + predict
                        + gp.Scan(scan_request)
                    )
                elif (
                    meta_data.num_samples == 0 and meta_data.num_channels != 0
                ):
                    pipeline = (
                        NapariImageSource(
                            raw,
                            raw_key,
                            gp.ArraySpec(
                                gp.Roi(
                                    (0,) * num_spatial_dims,
                                    raw.data.shape[-num_spatial_dims:],
                                ),
                                voxel_size=voxel_size,
                            ),
                        )
                        + gp.Pad(raw_key, context)
                        + gp.Unsqueeze([raw_key], 0)
                        + predict
                        + gp.Scan(scan_request)
                    )

                elif (
                    meta_data.num_samples != 0 and meta_data.num_channels == 0
                ):
                    pipeline = (
                        NapariImageSource(
                            raw,
                            raw_key,
                            gp.ArraySpec(
                                gp.Roi(
                                    (0,) * num_spatial_dims,
                                    raw.data.shape[-num_spatial_dims:],
                                ),
                                voxel_size=voxel_size,
                            ),
                        )
                        + gp.Pad(raw_key, context)
                        + gp.Unsqueeze([raw_key], 1)
                        + predict
                        + gp.Scan(scan_request)
                    )

                elif (
                    meta_data.num_samples != 0 and meta_data.num_channels != 0
                ):
                    pipeline = (
                        NapariImageSource(
                            raw,
                            raw_key,
                            gp.ArraySpec(
                                gp.Roi(
                                    (0,) * num_spatial_dims,
                                    raw.data.shape[-num_spatial_dims:],
                                ),
                                voxel_size=voxel_size,
                            ),
                        )
                        + gp.Pad(raw_key, context)
                        + predict
                        + gp.Scan(scan_request)
                    )
                # request to pipeline for ROI of whole image/volume
                request = gp.BatchRequest()
                request.add(raw_key, raw.data.shape[-num_spatial_dims:])
                request.add(prediction_key, raw.data.shape[-num_spatial_dims:])
                with gp.build(pipeline):
                    batch = pipeline.request_batch(request)

                prediction = batch.arrays[prediction_key].data
                colormaps = ["red", "green", "blue"]
                prediction_layers = [
                    (
                        prediction[:, i : i + 1, ...].copy(),
                        {
                            "name": "offset-"
                            + "zyx"[meta_data.num_spatial_dims - i]
                            if i < meta_data.num_spatial_dims
                            else "std",
                            "colormap": colormaps[
                                meta_data.num_spatial_dims - i
                            ]
                            if i < meta_data.num_spatial_dims
                            else "gray",
                            "blending": "additive",
                        },
                        "image",
                    )
                    for i in range(meta_data.num_spatial_dims + 1)
                ]

                labels = np.zeros_like(
                    prediction[:, 0:1, ...].data, dtype=np.uint64
                )

                for sample in tqdm(range(prediction.data.shape[0])):
                    embeddings = prediction[sample]
                    embeddings_std = embeddings[-1, ...]
                    embeddings_mean = embeddings[
                        np.newaxis, :num_spatial_dims, ...
                    ]
                    segmentation = mean_shift_segmentation(
                        embeddings_mean,
                        embeddings_std,
                        bandwidth=bandwidth,
                        min_size=min_size,
                    )
                    labels[
                        sample,
                        0,
                        ...,
                    ] = segmentation
                return prediction_layers + [
                    (labels, {"name": "Segmentation"}, "labels")
                ]

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

    @property
    def save_widget(self):
        # TODO: block buttons on call. This shouldn't take long, but other operations such
        # as continuing to train should be blocked until this is done.
        def on_return():
            self.set_buttons("paused")

        @magic_factory(call_button="Save")
        def save(path: Path = Path("checkpoint.pt")) -> FunctionWorker[None]:
            @thread_worker(
                connect={"returned": lambda: on_return()},
                progress={"total": 0, "desc": "Saving"},
            )
            def async_save(path: Path = Path("checkpoint.pt")) -> None:
                model, optimizer, scheduler = get_training_state()
                training_stats = get_training_stats()
                torch.save(
                    (
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                        training_stats,
                    ),
                    path,
                )

            return async_save(path)

        if not hasattr(self, "__save_widget"):
            self.__save_widget = save()

        return self.__save_widget

    @property
    def load_widget(self):
        # TODO: block buttons on call. This shouldn't take long, but other operations such
        # as continuing to train should be blocked until this is done.
        def on_return():
            self.update_progress_plot()
            self.set_buttons("paused")

        @magic_factory(call_button="Load")
        def load(path: Path = Path("checkpoint.pt")) -> FunctionWorker[None]:
            @thread_worker(
                connect={"returned": on_return},
                progress={"total": 0, "desc": "Saving"},
            )
            def async_load(path: Path = Path("checkpoint.pt")) -> None:
                model, optimizer, scheduler = get_training_state()
                training_stats = get_training_stats()
                state_dicts = torch.load(
                    path,
                )
                model.load_state_dict(state_dicts[0])
                optimizer.load_state_dict(state_dicts[1])
                scheduler.load_state_dict(state_dicts[2])
                training_stats.load(state_dicts[3])

            return async_load(path)

        if not hasattr(self, "__load_widget"):
            self.__load_widget = load()

        return self.__load_widget

    @property
    def training(self) -> bool:
        try:
            return self.__training
        except AttributeError:
            return False

    @training.setter
    def training(self, training: bool):
        self.__training = training
        if training:
            if self.__training_generator is None:
                self.start_training_loop()
            assert self.__training_generator is not None
            self.__training_generator.resume()
            self.set_buttons("training")
        else:
            if self.__training_generator is not None:
                self.__training_generator.send("stop")
                # button state handled by on_return

    def reset_training_state(self, keep_stats=False):
        if self.__training_generator is not None:
            self.__training_generator.quit()
        self.__training_generator = None
        if not keep_stats:
            training_stats = get_training_stats()
            training_stats.reset()
            if self.loss_plot is None:
                self.loss_plot = self.progress_plot.axes.plot(
                    [],
                    [],
                    label="Training Loss",
                )[0]
                self.progress_plot.axes.legend()
                self.progress_plot.axes.set_title("Training Progress")
                self.progress_plot.axes.set_xlabel("Iterations")
                self.progress_plot.axes.set_ylabel("Loss")
            self.update_progress_plot()

    def update_progress_plot(self):
        training_stats = get_training_stats()
        self.loss_plot.set_xdata(training_stats.iterations)
        self.loss_plot.set_ydata(training_stats.losses)
        self.progress_plot.axes.relim()
        self.progress_plot.axes.autoscale_view()
        with contextlib.suppress(np.linalg.LinAlgError):
            # matplotlib seems to throw a LinAlgError on draw sometimes. Not sure
            # why yet. Seems to only happen when initializing models without any
            # layers loaded. No idea whats going wrong.
            # For now just avoid drawing. Seems to work as soon as there is data to plot
            self.progress_plot.draw()

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

    def start_training_loop(self):
        self.reset_training_state(keep_stats=True)
        training_stats = get_training_stats()

        self.__training_generator = self.train_cellulus(
            self.raw_selector.value,
            self.get_selected_axes(),
            iteration=training_stats.iteration,
        )
        self.__training_generator.yielded.connect(self.on_yield)
        self.__training_generator.returned.connect(self.on_return)
        self.__training_generator.start()

    def train(self):
        self.training = not self.training

    def snapshot(self):
        self.__training_generator.send("snapshot")
        self.training = True

    def spatial_dims(self, ndims):
        return ["time", "z", "y", "x"][-ndims:]

    def create_train_widget(self, viewer):
        # inputs:
        raw = layer_choice_widget(
            viewer,
            annotation=napari.layers.Image,
            name="raw",
        )
        train_widget = Container(widgets=[raw])

        return train_widget

    def on_yield(self, step_data):
        iteration, loss, *layers = step_data
        if len(layers) > 0:
            self.add_layers(layers)
        if iteration is not None and loss is not None:
            training_stats = get_training_stats()
            training_stats.iteration = iteration
            training_stats.iterations.append(iteration)
            training_stats.losses.append(loss)
            self.update_progress_plot()

    def on_return(self, weights_path: Path):
        """
        Update model to use provided returned weights
        """
        global _model
        global _optimizer
        global _scheduler
        assert (
            _model is not None
            and _optimizer is not None
            and _scheduler is not None
        )
        model_state_dict, optim_state_dict, scheduler_state_dict = torch.load(
            weights_path
        )
        _model.load_state_dict(model_state_dict)
        _optimizer.load_state_dict(optim_state_dict)
        _scheduler.load_state_dict(scheduler_state_dict)
        self.reset_training_state(keep_stats=True)
        self.set_buttons("paused")

    def add_layers(self, layers):
        viewer_axis_labels = self.viewer.dims.axis_labels

        for data, metadata, layer_type in layers:
            # then try to update the viewer layer with that name.
            name = metadata.pop("name")
            axes = metadata.pop("axes")
            overwrite = metadata.pop("overwrite", False)
            slices = metadata.pop("slices", None)
            shape = metadata.pop("shape", None)

            # handle viewer axes if still default numerics
            # TODO: Support using xarray axis labels as soon as napari does
            if len(set(viewer_axis_labels).intersection(set(axes))) == 0:
                spatial_axes = [
                    axis for axis in axes if axis not in ["batch", "channel"]
                ]
                assert (
                    len(viewer_axis_labels) - len(spatial_axes) <= 1
                ), f"Viewer has axes: {viewer_axis_labels}, but we expect ((channels), {spatial_axes})"
                viewer_axis_labels = (
                    ("channels", *spatial_axes)
                    if len(viewer_axis_labels) > len(spatial_axes)
                    else spatial_axes
                )
                self.viewer.dims.axis_labels = viewer_axis_labels

            batch_dim = axes.index("batch") if "batch" in axes else -1
            assert batch_dim in [
                -1,
                0,
            ], "Batch dim must be first"
            if batch_dim == 0:
                data = data[0]

            if slices is not None and shape is not None:
                # strip channel dimension from slices and shape
                slices = (slice(None, None), *slices[1:])
                shape = (data.shape[0], *shape[1:])

                # create new data array with filled in chunk
                full_data = np.zeros(shape, dtype=data.dtype)
                full_data[slices] = data

            else:
                slices = tuple(slice(None, None) for _ in data.shape)
                full_data = data

            try:
                # add to existing layer
                layer = self.viewer.layers[name]

                if overwrite:
                    layer.data[slices] = data
                    layer.refresh()
                else:
                    # concatenate along batch dimension
                    layer.data = np.concatenate(
                        [
                            layer.data.reshape(-1, *full_data.shape),
                            full_data.reshape(-1, *full_data.shape).astype(
                                layer.data.dtype
                            ),
                        ],
                        axis=0,
                    )
                # make first dimension "batch" if it isn't
                if not overwrite and viewer_axis_labels[0] != "batch":
                    viewer_axis_labels = ("batch", *viewer_axis_labels)
                    self.viewer.dims.axis_labels = viewer_axis_labels

            except KeyError:  # layer not in the viewer
                # TODO: Support defining layer axes as soon as napari does
                if layer_type == "image":
                    self.viewer.add_image(full_data, name=name, **metadata)
                elif layer_type == "labels":
                    self.viewer.add_labels(
                        full_data.astype(int), name=name, **metadata
                    )

    @thread_worker
    def train_cellulus(
        self,
        raw,
        axis_names,
        iteration=0,
    ):

        train_config = get_train_config()

        # Turn layer into dataset:
        train_dataset = NapariDataset(
            raw,
            axis_names,
            crop_size=train_config.crop_size,
            control_point_spacing=train_config.control_point_spacing,
            control_point_jitter=train_config.control_point_jitter,
        )
        model, optimizer, scheduler = get_training_state(train_dataset)

        model.train()

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=train_config.batch_size,
            drop_last=True,
            num_workers=train_config.num_workers,
            pin_memory=True,
        )

        # set loss
        criterion = get_loss(
            regularizer_weight=train_config.regularizer_weight,
            temperature=train_config.temperature,
            kappa=train_config.kappa,
            density=train_config.density,
            num_spatial_dims=train_dataset.get_num_spatial_dims(),
            reduce_mean=train_config.reduce_mean,
            device=train_config.device,
        )

        def train_iteration(batch, model, criterion, optimizer, device):
            prediction = model(batch.to(device))
            loss = criterion(prediction)
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item(), prediction

        mode = yield (None, None)
        # call `train_iteration`
        for iteration, batch in tqdm(
            zip(
                range(iteration, train_config.max_iterations),
                train_dataloader,
            )
        ):
            train_loss, prediction = train_iteration(
                batch.float(),
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=train_config.device,
            )
            scheduler.step()

            if mode is None:
                mode = yield (
                    iteration,
                    train_loss,
                )

            elif mode == "stop":
                checkpoint = Path(f"/tmp/checkpoints/{iteration}.pt")
                if not checkpoint.parent.exists():
                    checkpoint.parent.mkdir(parents=True)
                torch.save(
                    (
                        model.state_dict(),
                        optimizer.state_dict(),
                        scheduler.state_dict(),
                    ),
                    checkpoint,
                )
                return checkpoint

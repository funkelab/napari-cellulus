"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING, Optional, List
import time

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from cellulus.configs.model_config import ModelConfig
from cellulus.configs.train_config import TrainConfig
from cellulus.models import get_model
from cellulus.criterions import get_loss
from cellulus.utils.mean_shift import mean_shift_segmentation


# local package imports
from copy import deepcopy
from .gui_helpers import layer_choice_widget, MplCanvas
from .dataset import NapariDataset
from .gp.nodes import NapariImageSource

# github repo libraries
import gunpowder as gp

# pip installed libraries
import napari
from napari.qt.threading import FunctionWorker, thread_worker
import torch
import numpy as np

# widget stuff
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from magicgui.widgets import create_widget, Container, Label
from superqt import QCollapsible
from qtpy.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QFileDialog,
    QInputDialog,
    QLabel,
    QFrame,
)

# python built in libraries
from pathlib import Path
from contextlib import contextmanager
import dataclasses
from tqdm import tqdm

if TYPE_CHECKING:
    import napari


################################## GLOBALS ####################################
_train_config: Optional[TrainConfig] = None
_model_config: Optional[ModelConfig] = None
_model: Optional[torch.nn.Module] = None
_optimizer: Optional[torch.optim.Optimizer] = None
_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None


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


@magic_factory(call_button="Save")
def train_config_widget(
    crop_size: list[int] = list([256, 256]),
    batch_size: int = 8,
    max_iterations: int = 100_000,
    initial_learning_rate: float = 4e-5,
    density: float = 0.2,
    kappa: float = 10.0,
    temperature: float = 10.0,
    regularizer_weight: float = 1e-5,
    reduce_mean: bool = True,
    save_model_every: int = 1_000,
    save_snapshot_every: int = 1_000,
    num_workers: int = 8,
    control_point_spacing: int = 64,
    control_point_jitter: float = 2.0,
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
    downsampling_factors: list[list[int]] = list([list([2, 2])]),
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
    if _model_config is None:
        # TODO: deal with hard coded defaults
        _model_config = ModelConfig(num_fmaps=24, fmap_inc_factor=3)
    if _train_config is None:
        _train_config = get_train_config()
    if _model is None:
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
        ).cuda()

        for name, layer in _model.named_modules():
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


@magic_factory(call_button="Predict")
def predict(
    raw: napari.layers.Image,
    crop_size: list[int] = list([252, 252]),
    p_salt_pepper: float = 0.1,
    num_infer_iterations: int = 16,
) -> FunctionWorker[List[napari.types.LayerDataTuple]]:
    # TODO: do this better?
    @thread_worker(
        connect={"returned": lambda: None},
        progress={"total": 0, "desc": "Predicting"},
    )
    def async_predict(
        raw: napari.layers.Image,
        crop_size: list[int],
        p_salt_pepper: float,
        num_infer_iterations: int,
    ) -> List[napari.types.LayerDataTuple]:
        raw.data = raw.data.astype(np.float32)

        global _model
        assert (
            _model is not None
        ), "You must train a model before running inference"
        model = _model

        num_spatial_dims = len(raw.data.shape) - 2
        num_channels = raw.data.shape[1]

        voxel_size = gp.Coordinate((1,) * num_spatial_dims)
        model.set_infer(
            p_salt_pepper=p_salt_pepper,
            num_infer_iterations=num_infer_iterations,
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
                ).cuda()
            ).shape
        )

        input_size = gp.Coordinate(input_shape[2:]) * voxel_size
        output_size = gp.Coordinate(output_shape[2:]) * voxel_size

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
            array_specs={prediction_key: gp.ArraySpec(voxel_size=voxel_size)},
        )

        pipeline = (
            NapariImageSource(
                raw,
                raw_key,
                gp.ArraySpec(
                    gp.Roi((0,) * num_spatial_dims, raw.data.shape[2:]),
                    voxel_size=voxel_size,
                ),
            )
            + gp.Pad(raw_key, context)
            + predict
            + gp.Scan(scan_request)
        )

        # request to pipeline for ROI of whole image/volume
        request = gp.BatchRequest()
        request.add(raw_key, raw.data.shape[2:])
        request.add(prediction_key, raw.data.shape[2:])
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)

        prediction = batch.arrays[prediction_key].data
        colormaps = ["red", "green", "blue"]
        prediction = [
            (
                prediction[:, i : i + 1, ...],
                {
                    "name": "offset-" + "zyx"[num_channels - i]
                    if i < num_channels
                    else "std",
                    "colormap": colormaps[num_channels - i]
                    if i < num_channels
                    else "gray",
                    "blending": "additive",
                },
                "image",
            )
            for i in range(num_channels + 1)
        ]
        return prediction

    return async_predict(
        raw,
        crop_size=crop_size,
        p_salt_pepper=p_salt_pepper,
        num_infer_iterations=num_infer_iterations,
    )


@magic_factory(call_button="Segment")
def segment(
    offset_layers: list[napari.layers.Image],
    bandwidth: int = 7,
    min_size: int = 10,
) -> FunctionWorker[napari.types.LayerDataTuple]:
    @thread_worker(
        connect={"returned": lambda: None},
        progress={"total": 0, "desc": "Segmenting"},
    )
    def async_segment(
        offset_layers: list[napari.layers.Image],
        bandwidth: int = 7,
        min_size: int = 10,
    ) -> napari.types.LayerDataTuple:
        labels = np.zeros_like(offset_layers[0].data, dtype=np.uint64)
        num_spatial_dims = len(offset_layers[0].data.shape) - 2

        for sample in tqdm(range(offset_layers[0].data.shape[0])):
            embeddings = np.concatenate(
                [layer.data[sample] for layer in offset_layers]
            )
            embeddings_std = embeddings[-1, ...]
            embeddings_mean = embeddings[np.newaxis, :num_spatial_dims, ...]
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
        return (labels, {"name": "Segmentation"}, "labels")

    return async_segment(
        offset_layers,
        bandwidth,
        min_size,
    )


@magic_factory(call_button="Save")
def save(path: Path = Path("checkpoint.pt")) -> FunctionWorker[None]:
    @thread_worker(
        connect={"returned": lambda: None},
        progress={"total": 0, "desc": "Saving"},
    )
    def async_save(path: Path = Path("checkpoint.pt")) -> None:
        model, optimizer, scheduler = get_training_state()
        torch.save(
            (
                model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
            ),
            path,
        )

    return async_save(path)


@magic_factory(call_button="Load")
def load(path: Path = Path("checkpoint.pt")) -> FunctionWorker[None]:
    @thread_worker(
        connect={"returned": lambda: None},
        progress={"total": 0, "desc": "Saving"},
    )
    def async_load(path: Path = Path("checkpoint.pt")) -> None:
        model, optimizer, scheduler = get_training_state()
        state_dicts = torch.load(
            path,
        )
        model.load_state_dict(state_dicts[0])
        optimizer.load_state_dict(state_dicts[1])
        scheduler.load_state_dict(state_dicts[2])

    return async_load(path)


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

        # add buttons
        self.train_button = QPushButton("Train!", self)
        self.train_button.clicked.connect(self.train)
        layout.addWidget(self.train_button)
        self.predict_button = QPushButton("Predict!", self)
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

        # activate layout
        self.setLayout(layout)

        # Widget state
        self.model = None
        self.reset_training_state()

        # No buttons should be enabled
        self.disable_buttons()

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
            self.train_button.setText("Pause!")
            self.disable_buttons()
        else:
            if self.__training_generator is not None:
                self.__training_generator.send("stop")
            self.train_button.setText("Train!")
            self.disable_buttons()

    def reset_training_state(self, keep_stats=False):
        if self.__training_generator is not None:
            self.__training_generator.quit()
        self.__training_generator = None
        if not keep_stats:
            self.iteration = 0
            self.__iterations = []
            self.__losses = []
            self.__val_iterations = []
            self.__val_losses = []
            if self.loss_plot is None:
                self.loss_plot = self.progress_plot.axes.plot(
                    self.__iterations, self.__losses, label="Training Loss"
                )[0]
                self.val_plot = self.progress_plot.axes.plot(
                    self.__val_iterations,
                    self.__val_losses,
                    label="Validation Loss",
                )[0]
                self.progress_plot.axes.legend()
                if self.model is not None:
                    self.progress_plot.axes.set_title(
                        f"{self.model.name} Training Progress"
                    )
                else:
                    self.progress_plot.axes.set_title(f"Training Progress")
                self.progress_plot.axes.set_xlabel("Iterations")
                self.progress_plot.axes.set_ylabel("Loss")
            self.update_progress_plot()

    def update_progress_plot(self):
        self.loss_plot.set_xdata(self.__iterations)
        self.loss_plot.set_ydata(self.__losses)
        self.val_plot.set_xdata(self.__val_iterations)
        self.val_plot.set_ydata(self.__val_losses)
        self.progress_plot.axes.relim()
        self.progress_plot.axes.autoscale_view()
        try:
            self.progress_plot.draw()
        except np.linalg.LinAlgError as e:
            # matplotlib seems to throw a LinAlgError on draw sometimes. Not sure
            # why yet. Seems to only happen when initializing models without any
            # layers loaded. No idea whats going wrong.
            # For now just avoid drawing. Seems to work as soon as there is data to plot
            pass

    def disable_buttons(
        self,
        train: bool = False,
    ):
        self.train_button.setEnabled(not train)

    def start_training_loop(self):
        self.reset_training_state(keep_stats=True)

        self.__training_generator = self.train_cellulus(
            self.raw_selector.value,
            iteration=self.iteration,
        )
        self.__training_generator.yielded.connect(self.on_yield)
        self.__training_generator.returned.connect(self.on_return)
        self.__training_generator.start()

        # all buttons are enabled while the training loop is running
        self.disable_buttons()

    def train(self):
        self.training = not self.training

    def predict(self):
        if self.training:
            self.__training_generator.send("predict")
        else:
            raise NotImplementedError()

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
            self.iteration = iteration
            self.__iterations.append(iteration)
            self.__losses.append(loss)
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
        self.disable_buttons()

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
            ], f"Batch dim must be first"
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
        iteration=0,
    ):
        train_config = get_train_config()
        # Turn layer into dataset:
        train_dataset = NapariDataset(
            raw,
            crop_size=train_config.crop_size,
            control_point_spacing=train_config.control_point_spacing,
            control_point_jitter=train_config.control_point_jitter,
        )
        model, optimizer, scheduler = get_training_state(train_dataset)

        # TODO: How to display profiling stats
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device)
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
        )

        def train_iteration(
            batch,
            model,
            criterion,
            optimizer,
        ):
            prediction = model(batch.cuda())
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
            )
            scheduler.step()

            if mode is None:
                mode = yield (
                    iteration,
                    train_loss,
                )
            elif mode == "predict":
                prediction = model(torch.from_numpy(raw.data).cuda().float())

                # Generate affinities and keep the offsets as metadata
                prediction_layers = [
                    (
                        prediction.detach().cpu().numpy(),
                        {
                            "name": "Cellulus(online)",
                            "axes": (
                                "sample",
                                "channel",
                                "y",
                                "x",
                            ),
                        },
                        "image",
                    ),
                ]
                mode = yield (iteration, train_loss, *prediction_layers)

            # if iteration % train_config.save_model_every == 0:
            #     is_lowest = train_loss < lowest_loss
            #     lowest_loss = min(train_loss, lowest_loss)
            #     state = {
            #         "iteration": iteration,
            #         "lowest_loss": lowest_loss,
            #         "model_state_dict": model.state_dict(),
            #         "optim_state_dict": optimizer.state_dict(),
            #         "logger_data": logger.data,
            #     }
            #     save_model(state, iteration, is_lowest)

            # if iteration % train_config.save_snapshot_every == 0:
            #     save_snapshot(
            #         batch,
            #         prediction,
            #         iteration,
            #     )
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

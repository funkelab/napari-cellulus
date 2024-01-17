import time
from pathlib import Path

import gunpowder as gp
import napari
import numpy as np
import pyqtgraph as pg
import torch
from cellulus.configs.inference_config import InferenceConfig
from cellulus.configs.model_config import ModelConfig
from cellulus.configs.train_config import TrainConfig
from cellulus.criterions import get_loss
from cellulus.models import get_model
from cellulus.train import train_iteration
from cellulus.utils.mean_shift import mean_shift_segmentation

# widget stuff
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from scipy.ndimage import distance_transform_edt as dtedt
from skimage.filters import threshold_otsu
from superqt import QCollapsible, QLabeledDoubleSlider
from torch import nn

from ..dataset import NapariDataset
from ..gp.nodes.napari_image_source import NapariImageSource

# local package imports
from ..gui_helpers import layer_choice_widget

############ GLOBALS ###################
time_now = 0
train_config = None
model_config = None
segment_config = None
model = None
optimizer = None
scheduler = None
dataset = None


class Model(nn.Module):
    def __init__(self, model, selected_axes):
        super().__init__()
        self.model = model
        self.selected_axes = selected_axes

    def forward(self, x):
        if "s" in self.selected_axes and "c" in self.selected_axes:
            pass
        elif "s" in self.selected_axes and "c" not in self.selected_axes:
            x = torch.unsqueeze(x, 1)
        elif "s" not in self.selected_axes and "c" in self.selected_axes:
            pass
        elif "s" not in self.selected_axes and "c" not in self.selected_axes:
            x = torch.unsqueeze(x, 1)

        return self.model(x)

    def set_infer(self, p_salt_pepper, num_infer_iterations, device):
        self.model.eval()
        self.model.set_infer(p_salt_pepper, num_infer_iterations, device)


class SegmentationWidget(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()
        self.widget = QWidget()
        self.scroll = QScrollArea()
        self.viewer = napari_viewer

        # Initialize `train_config` and `model_config`

        self.train_config = None
        self.model_config = None
        self.segment_config = None

        # Initialize losses and iterations
        self.losses = []
        self.iterations = []

        # initialize mode. this will change to 'training', 'predicting' and 'segmenting' later
        self.mode = "configuring"

        # initialize UI components
        text_label = QLabel("<h3>Cellulus</h3>")

        self.method_description_label = QLabel(
            '<small>Unsupervised Learning of Object-Centric Embeddings<br>for Cell Instance Segmentation in Microscopy Images.<br>If you are using this in your research, please <a href="https://github.com/funkelab/cellulus#citation" style="color:gray;">cite us</a>.</small><br><small><tt><a href="https://github.com/funkelab/cellulus" style="color:gray;">https://github.com/funkelab/cellulus</a></tt></small>'
        )
        # inner layout
        grid_0 = QGridLayout()
        grid_0.addWidget(text_label, 1, 0, 1, 1)
        grid_0.addWidget(self.method_description_label, 0, 1, 2, 1)

        # Initialize object size widget
        object_size_label = QLabel(self)
        object_size_label.setText("Object Width [px]:")
        self.object_size_line = QLineEdit(self)
        self.object_size_line.setText("30")

        device_label = QLabel(self)
        device_label.setText("Device")
        self.device_combo_box = QComboBox(self)
        self.device_combo_box.addItem("cpu")
        self.device_combo_box.addItem("cuda:0")
        self.device_combo_box.addItem("mps")
        self.device_combo_box.setCurrentText("mps")

        grid_1 = QGridLayout()
        grid_1.addWidget(object_size_label, 0, 0, 1, 1)
        grid_1.addWidget(self.object_size_line, 0, 1, 1, 1)
        grid_1.addWidget(device_label, 1, 0, 1, 1)
        grid_1.addWidget(self.device_combo_box, 1, 1, 1, 1)

        # Initialize choice of layer.
        self.raw_selector = layer_choice_widget(
            self.viewer, annotation=napari.layers.Image, name="raw"
        )

        # Initialize Checkboxes
        self.s_checkbox = QCheckBox("s")
        self.c_checkbox = QCheckBox("c")
        self.z_checkbox = QCheckBox("z")
        self.y_checkbox = QCheckBox("y")
        self.x_checkbox = QCheckBox("x")

        axis_layout = QHBoxLayout()
        axis_layout.addWidget(self.s_checkbox)
        axis_layout.addWidget(self.c_checkbox)
        axis_layout.addWidget(self.z_checkbox)
        axis_layout.addWidget(self.y_checkbox)
        axis_layout.addWidget(self.x_checkbox)
        axis_selector = QGroupBox("Axis Names:")
        axis_selector.setLayout(axis_layout)

        if self.raw_selector.value is not None:
            self.update_axis_layout()
        self.raw_selector.native.currentTextChanged.connect(
            self.update_axis_layout
        )

        # Initialize train configs widget
        crop_size_label = QLabel(self)
        crop_size_label.setText("Crop Size")
        self.crop_size_line = QLineEdit(self)
        self.crop_size_line.setText("252")
        batch_size_label = QLabel(self)
        batch_size_label.setText("Batch Size")
        self.batch_size_line = QLineEdit(self)
        self.batch_size_line.setText("8")
        max_iterations_label = QLabel(self)
        max_iterations_label.setText("Max iterations")
        self.max_iterations_line = QLineEdit(self)
        self.max_iterations_line.setText("2000")

        grid_2 = QGridLayout()
        grid_2.addWidget(
            crop_size_label,
            0,
            0,
            1,
            1,
        )
        grid_2.addWidget(self.crop_size_line, 0, 1, 1, 1)
        grid_2.addWidget(
            batch_size_label,
            2,
            0,
            1,
            1,
        )
        grid_2.addWidget(self.batch_size_line, 2, 1, 1, 1)
        grid_2.addWidget(
            max_iterations_label,
            3,
            0,
            1,
            1,
        )
        grid_2.addWidget(self.max_iterations_line, 3, 1, 1, 1)

        train_configs_widget = QWidget()
        train_configs_widget.setLayout(grid_2)
        collapsible_train_configs = QCollapsible("Train Configs", self)
        collapsible_train_configs.addWidget(train_configs_widget)

        # Initialize model configs widget

        fmaps_label = QLabel(self)
        fmaps_label.setText("Number of feature maps")
        self.fmaps_line = QLineEdit(self)
        self.fmaps_line.setText("24")
        fmaps_increase_label = QLabel(self)
        fmaps_increase_label.setText("Feature maps Inc. factor")
        self.fmaps_increase_line = QLineEdit(self)
        self.fmaps_increase_line.setText("3")
        self.train_from_scratch_checkbox = QCheckBox("Train from scratch")

        grid_3 = QGridLayout()
        grid_3.addWidget(
            fmaps_label,
            0,
            0,
            1,
            1,
        )
        grid_3.addWidget(self.fmaps_line, 0, 1, 1, 1)
        grid_3.addWidget(
            fmaps_increase_label,
            2,
            0,
            1,
            1,
        )
        grid_3.addWidget(self.fmaps_increase_line, 2, 1, 1, 1)
        grid_3.addWidget(self.train_from_scratch_checkbox, 3, 0, 1, 1)

        model_configs_widget = QWidget()
        model_configs_widget.setLayout(grid_3)
        collapsible_model_configs = QCollapsible("Model Configs", self)
        collapsible_model_configs.addWidget(model_configs_widget)

        # Initialize loss/iterations widget

        self.losses_widget = pg.PlotWidget()
        self.losses_widget.setBackground((37, 41, 49))
        styles = {"color": "white", "font-size": "16px"}
        self.losses_widget.setLabel("left", "Loss", **styles)
        self.losses_widget.setLabel("bottom", "Iterations", **styles)

        # Initialize Train Button
        self.train_button = QPushButton("Train", self)
        self.train_button.clicked.connect(self.prepare_for_training)

        # Initialize Save and Load Widget
        collapsible_save_load_widget = QCollapsible(
            "Save and Load Model", self
        )

        # Initialize save/load model

        self.save_model_button = QPushButton(self)
        self.save_model_button.setText("Save Model")
        self.save_model_button.clicked.connect(self.save_model_weights)
        self.load_model_button = QPushButton(self)
        self.load_model_button.setText("Load Model")
        self.load_model_button.clicked.connect(self.get_model_weights)

        grid_4 = QGridLayout()
        grid_4.addWidget(self.save_model_button, 0, 0, 1, 1)
        grid_4.addWidget(self.load_model_button, 0, 1, 1, 1)
        save_load_widget = QWidget()
        save_load_widget.setLayout(grid_4)
        collapsible_save_load_widget.addWidget(save_load_widget)

        # Initialize Inference Configs widget
        crop_size_infer_label = QLabel(self)
        crop_size_infer_label.setText("Crop Size")
        self.crop_size_infer_line = QLineEdit(self)
        self.crop_size_infer_line.setText("252")
        num_infer_iterations_label = QLabel(self)
        num_infer_iterations_label.setText("No. of iterations")
        self.num_infer_iterations_line = QLineEdit(self)
        self.num_infer_iterations_line.setText("16")

        grid_5 = QGridLayout()
        grid_5.addWidget(crop_size_infer_label, 0, 0, 1, 1)
        grid_5.addWidget(self.crop_size_infer_line, 0, 1, 1, 1)
        grid_5.addWidget(num_infer_iterations_label, 1, 0, 1, 1)
        grid_5.addWidget(self.num_infer_iterations_line, 1, 1, 1, 1)
        inference_widget = QWidget()
        inference_widget.setLayout(grid_5)
        collapsible_inference_configs = QCollapsible("Inference Configs", self)
        collapsible_inference_configs.addWidget(inference_widget)

        # Initialize Predict Embeddings Button
        self.predict_embeddings_button = QPushButton(
            "Predict Embeddings", self
        )
        self.predict_embeddings_button.clicked.connect(
            self.prepare_for_prediction
        )

        bandwidth_label = QLabel(self)
        bandwidth_label.setText("Bandwidth")
        self.bandwidth_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.bandwidth_slider.setRange(0, 100)

        binary_threshold_label = QLabel(self)
        binary_threshold_label.setText("Threshold")
        self.binary_threshold_slider = QLabeledDoubleSlider(
            Qt.Orientation.Horizontal
        )
        self.binary_threshold_slider.setRange(0, 100)
        self.binary_threshold_slider.sliderReleased.connect(
            self.prepare_for_segmenting
        )

        grid_6 = QGridLayout()
        grid_6.addWidget(bandwidth_label, 0, 0, 1, 1)
        grid_6.addWidget(self.bandwidth_slider, 0, 1, 1, 1)
        grid_6.addWidget(binary_threshold_label, 1, 0, 1, 1)
        grid_6.addWidget(self.binary_threshold_slider, 1, 1, 1, 1)

        segmentation_widget = QWidget()
        segmentation_widget.setLayout(grid_6)
        collapsible_segmentation_configs = QCollapsible(
            "Segmentation Configs", self
        )
        collapsible_segmentation_configs.addWidget(segmentation_widget)

        # Initialize Feedback Button
        self.feedback_label = QLabel(
            '<small>Please share any feedback <a href="https://github.com/funkelab/napari-cellulus/issues/new/choose" style="color:gray;">here</a>.</small>'
        )

        # specify outer layout
        outer_layout = QVBoxLayout()

        # Add all components to outer_layout

        outer_layout.addLayout(grid_0)
        outer_layout.addWidget(self.raw_selector.native)
        outer_layout.addWidget(axis_selector)
        outer_layout.addLayout(grid_1)
        outer_layout.addWidget(collapsible_train_configs)
        outer_layout.addWidget(collapsible_model_configs)
        outer_layout.addWidget(self.losses_widget)
        outer_layout.addWidget(self.train_button)
        outer_layout.addWidget(collapsible_save_load_widget)
        outer_layout.addWidget(collapsible_inference_configs)
        outer_layout.addWidget(self.predict_embeddings_button)
        outer_layout.addWidget(collapsible_segmentation_configs)
        outer_layout.addWidget(self.feedback_label)
        outer_layout.setSpacing(20)
        self.widget.setLayout(outer_layout)

        self.scroll.setWidget(self.widget)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)

        self.setFixedWidth(400)
        self.setCentralWidget(self.scroll)

    def get_selected_axes(self):
        names = []
        for name, checkbox in zip(
            "sczyx",
            [
                self.s_checkbox,
                self.c_checkbox,
                self.z_checkbox,
                self.y_checkbox,
                self.x_checkbox,
            ],
        ):

            if checkbox.isChecked():
                names.append(name)
        return names

    def update_axis_layout(self):
        im_shape = self.raw_selector.value.data.shape
        if len(im_shape) == 2:
            self.y_checkbox.setChecked(True)
            self.x_checkbox.setChecked(True)
        elif len(im_shape) == 3:
            self.z_checkbox.setChecked(True)
            self.y_checkbox.setChecked(True)
            self.x_checkbox.setChecked(True)
        elif len(im_shape) == 4:
            self.c_checkbox.setChecked(True)
            self.z_checkbox.setChecked(True)
            self.y_checkbox.setChecked(True)
            self.x_checkbox.setChecked(True)
        elif len(im_shape) == 5:
            self.s_checkbox.setChecked(True)
            self.c_checkbox.setChecked(True)
            self.z_checkbox.setChecked(True)
            self.y_checkbox.setChecked(True)
            self.x_checkbox.setChecked(True)

    def get_model_weights(self):

        self.update_mode(self.sender())
        global model_config
        fname, _ = QFileDialog.getOpenFileName(
            self, "Browse to model weights", "models/last.pth"
        )
        model_config.checkpoint = fname

    def save_model_weights(self):

        global model, optimizer
        self.update_mode(self.sender())
        QFileDialog.getSaveFileName(self, "Save model weights")
        # if self.mode =='configuring':
        #    state = {
        #        "iteration": self.iterations[-1],
        #        "model_state_dict": model.state_dict(),
        #        "optim_state_dict": optimizer.state_dict(),
        #        "iterations": self.iterations,
        #        "losses": self.losses,
        #    }
        #    torch.save(state, filename)
        #    self.worker.quit()

    def prepare_for_training(self):

        global model, optimizer, model_config

        train_config = TrainConfig(
            crop_size=[
                int(self.crop_size_line.text())
            ],  # this is correctly handled in the dataset class
            batch_size=int(self.batch_size_line.text()),
            max_iterations=int(self.max_iterations_line.text()),
            device=self.device_combo_box.currentText(),
        )
        model_config = ModelConfig(
            num_fmaps=int(self.fmaps_line.text()),
            fmap_inc_factor=int(self.fmaps_increase_line.text()),
        )

        self.update_mode(self.sender())

        if self.mode == "training":
            self.worker = self.train(train_config, model_config)
            self.worker.yielded.connect(self.on_yield)
            self.worker.returned.connect(self.on_return)
            self.worker.start()
        elif self.mode == "configuring":
            state = {
                "iteration": self.iterations[-1],
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "iterations": self.iterations,
                "losses": self.losses,
            }
            filename = Path("models") / "last.pth"
            torch.save(state, filename)
            self.worker.quit()

    def update_mode(self, sender):

        if self.train_button.text() == "Train" and sender == self.train_button:
            self.train_button.setText("Pause")
            self.mode = "training"
            self.predict_embeddings_button.setEnabled(False)
            self.load_model_button.setEnabled(False)
        elif (
            self.train_button.text() == "Pause" and sender == self.train_button
        ):
            self.train_button.setText("Train")
            self.mode = "configuring"
            self.predict_embeddings_button.setEnabled(True)
            self.load_model_button.setEnabled(True)
        elif (
            self.predict_embeddings_button.text() == "Predict Embeddings"
            and sender == self.predict_embeddings_button
        ):
            self.predict_embeddings_button.setText("Pause")
            self.mode = "predicting"
            self.train_button.setEnabled(False)
            self.save_model_button.setEnabled(False)
            self.load_model_button.setEnabled(False)
        elif (
            self.predict_embeddings_button.text() == "Pause"
            and sender == self.predict_embeddings_button
        ):
            self.predict_embeddings_button.setText("Predict Embeddings")
            self.mode = "configuring"
            self.train_button.setEnabled(True)
            self.save_model_button.setEnabled(True)
            self.load_model_button.setEnabled(True)
        elif sender == self.binary_threshold_slider:
            self.mode = "segmenting"
            self.train_button.setEnabled(False)
            self.save_model_button.setEnabled(False)
            self.load_model_button.setEnabled(False)

    @thread_worker
    def train(self, train_config, model_config):

        global dataset, model, optimizer, scheduler

        # Turn layer into dataset
        dataset = NapariDataset(
            layer=self.raw_selector.value,
            axis_names=self.get_selected_axes(),
            crop_size=train_config.crop_size[0],  # list to integer
            control_point_spacing=train_config.control_point_spacing,
            control_point_jitter=train_config.control_point_jitter,
        )

        if not Path("models").exists():
            Path("models").mkdir()

        # create train dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=train_config.batch_size,
            drop_last=True,
            num_workers=train_config.num_workers,
            pin_memory=True,
        )

        if dataset.get_num_spatial_dims() == 2:
            downsampling_factors = [
                [2, 2],
            ]
        elif dataset.get_num_spatial_dims() == 3:
            downsampling_factors = [
                [2, 2, 2],
            ]

        # set model
        model_orig = get_model(
            in_channels=dataset.get_num_channels(),
            out_channels=dataset.get_num_spatial_dims(),
            num_fmaps=model_config.num_fmaps,
            fmap_inc_factor=model_config.fmap_inc_factor,
            features_in_last_layer=model_config.features_in_last_layer,
            downsampling_factors=[
                tuple(factor) for factor in downsampling_factors
            ],
            num_spatial_dims=dataset.get_num_spatial_dims(),
        )

        # put a wrapper around the model
        model = Model(model_orig, self.get_selected_axes())

        # set device
        device = torch.device(train_config.device)

        model = model.to(device)

        # initialize model weights
        if model_config.initialize:
            for _name, layer in model.named_modules():
                if isinstance(layer, torch.nn.modules.conv._ConvNd):
                    torch.nn.init.kaiming_normal_(
                        layer.weight, nonlinearity="relu"
                    )

        # set loss
        criterion = get_loss(
            regularizer_weight=train_config.regularizer_weight,
            temperature=train_config.temperature,
            kappa=train_config.kappa,
            density=train_config.density,
            num_spatial_dims=dataset.get_num_spatial_dims(),
            reduce_mean=train_config.reduce_mean,
            device=device,
        )

        # set optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.initial_learning_rate,
        )

        # set scheduler:

        def lambda_(iteration):
            return pow((1 - ((iteration) / train_config.max_iterations)), 0.9)

        # resume training
        if len(self.iterations) == 0:
            start_iteration = 0
        else:
            start_iteration = self.iterations[-1]

        if model_config.checkpoint is None:
            pass
        else:
            print(f"Resuming model from {model_config.checkpoint}")
            state = torch.load(model_config.checkpoint, map_location=device)
            start_iteration = state["iteration"] + 1
            self.iterations = state["iterations"]
            self.losses = state["losses"]
            model.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optim_state_dict"])

        # call `train_iteration`
        for iteration, batch in zip(
            range(start_iteration, train_config.max_iterations),
            dataloader,
        ):
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda_, last_epoch=iteration - 1
            )

            train_loss, oce_loss, prediction = train_iteration(
                batch,
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            scheduler.step()
            yield (iteration, train_loss)

    def on_yield(self, step_data):
        if self.mode == "training":
            global time_now
            iteration, loss = step_data
            current_time = time.time()
            time_elapsed = current_time - time_now
            time_now = current_time
            print(
                f"iteration {iteration}, loss {loss}, seconds/iteration {time_elapsed}"
            )
            self.iterations.append(iteration)
            self.losses.append(loss)
            self.update_canvas()
        elif self.mode == "predicting":
            print(step_data)
            # self.pbar.setValue(step_data)

    def update_canvas(self):
        self.losses_widget.plot(self.iterations, self.losses)

    def on_return(self, layers):
        # Describes what happens once worker is completed

        for data, metadata, layer_type in layers:
            if layer_type == "image":
                self.viewer.add_image(data, **metadata)
            elif layer_type == "labels":
                self.viewer.add_labels(data.astype(int), **metadata)
        self.update_mode(self.predict_embeddings_button)
        self.worker.quit()

    def prepare_for_prediction(self):
        global inference_config

        inference_config = InferenceConfig(
            crop_size=[int(self.crop_size_infer_line.text())],
            num_infer_iterations=int(self.num_infer_iterations_line.text()),
        )

        # update mode
        self.update_mode(self.sender())

        if self.mode == "predicting":
            self.worker = self.predict()
            self.worker.yielded.connect(self.on_yield)
            self.worker.returned.connect(self.on_return)
            self.worker.start()
        elif self.mode == "configuring":
            self.worker.quit()

    def prepare_for_segmenting(self):
        global inference_config

        print("=" * 10)
        print(self.sender())
        print(self.mode)

        # update mode
        self.update_mode(self.sender())

        if self.mode == "segmenting":
            self.worker = self.segment()
            self.worker.start()

    @thread_worker
    def predict(self):
        global inference_config, model, dataset

        raw_image = self.raw_selector.value

        if inference_config.bandwidth is None:
            inference_config.bandwidth = int(
                0.5 * float(self.object_size_line.text())
            )
        if inference_config.min_size is None:
            inference_config.min_size = int(
                0.1 * np.pi * (float(self.object_size_line.text()) ** 2) / 4
            )

        device = torch.device(self.device_combo_box.currentText())
        model.set_infer(
            p_salt_pepper=inference_config.p_salt_pepper,
            num_infer_iterations=inference_config.num_infer_iterations,
            device=device,
        )

        # check if dataset is not None.
        # user might directly go to inference ...

        num_spatial_dims = dataset.num_spatial_dims
        num_channels = dataset.num_channels
        spatial_dims = dataset.spatial_dims
        num_samples = dataset.num_samples
        num_dims = dataset.num_dims
        num_channels_temp = 1 if num_channels == 0 else num_channels

        crop_size = (inference_config.crop_size[0],) * num_spatial_dims

        input_shape = gp.Coordinate((1, num_channels_temp, *crop_size))
        if num_channels == 0:
            output_shape = gp.Coordinate(
                model(
                    torch.zeros((1, *crop_size), dtype=torch.float32).to(
                        device
                    )
                ).shape
            )
        else:
            output_shape = gp.Coordinate(
                model(
                    torch.zeros(
                        (1, num_channels_temp, *crop_size), dtype=torch.float32
                    ).to(device)
                ).shape
            )

        voxel_size = (1,) * (num_spatial_dims)
        model.eval()
        model.set_infer(
            p_salt_pepper=inference_config.p_salt_pepper,
            num_infer_iterations=inference_config.num_infer_iterations,
            device=device,
        )

        input_size = gp.Coordinate(input_shape[2:]) * gp.Coordinate(voxel_size)
        output_size = gp.Coordinate(output_shape[2:]) * gp.Coordinate(
            voxel_size
        )

        context = (input_size - output_size) / 2

        raw = gp.ArrayKey("RAW")
        prediction = gp.ArrayKey("PREDICT")

        scan_request = gp.BatchRequest()
        scan_request.add(raw, input_size)
        scan_request.add(prediction, output_size)

        predict = gp.torch.Predict(
            model,
            inputs={"x": raw},
            outputs={0: prediction},
            array_specs={prediction: gp.ArraySpec(voxel_size=voxel_size)},
        )

        pipeline = NapariImageSource(
            raw_image,
            raw,
            gp.ArraySpec(
                gp.Roi(
                    (0,) * num_dims,
                    raw_image.data.shape,
                ),
                voxel_size=(1,) * num_dims,
            ),
            spatial_dims,
        )
        if num_samples == 0:
            pipeline += (
                gp.Pad(raw, context, mode="reflect")
                + gp.Unsqueeze([raw], 0)
                + predict
                + gp.Scan(scan_request)
            )
        else:
            pipeline += (
                gp.Pad(raw, context, mode="reflect")
                + predict
                + gp.Scan(scan_request)
            )

        # request to pipeline for ROI of whole image/volume
        request = gp.BatchRequest()
        request.add(prediction, raw_image.data.shape[-num_spatial_dims:])

        counter = 0
        with gp.build(pipeline):
            batch = pipeline.request_batch(request)
            yield counter
            counter += 0.1

        prediction_data = batch.arrays[prediction].data
        print(f"prediction data has shape{prediction_data.shape}")

        colormaps = ["red", "green", "blue"]
        prediction_layers = [
            (
                prediction_data[:, i : i + 1, ...].copy(),
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

        foreground = np.zeros_like(prediction_data[:, 0:1, ...], dtype=bool)
        for sample in range(prediction_data.shape[0]):
            embeddings = prediction_data[sample]
            embeddings_std = embeddings[-1, ...]
            threshold = threshold_otsu(embeddings_std)
            print(f"Threshold for sample {sample} is {threshold}")
            binary_mask = embeddings_std < threshold
            foreground[sample, 0, ...] = binary_mask

        labels = np.zeros_like(prediction_data[:, 0:1, ...], dtype=np.uint64)
        for sample in range(prediction_data.shape[0]):
            embeddings = prediction_data[sample]
            embeddings_std = embeddings[-1, ...]
            embeddings_mean = embeddings[np.newaxis, :num_spatial_dims, ...]
            threshold = threshold_otsu(embeddings_std)
            segmentation = mean_shift_segmentation(
                embeddings_mean,
                embeddings_std,
                inference_config.bandwidth,
                inference_config.min_size,
                inference_config.reduction_probability,
                threshold,
            )
            labels[sample, 0, ...] = segmentation

        pp_labels = np.zeros_like(
            prediction_data[:, 0:1, ...], dtype=np.uint64
        )
        for sample in range(prediction_data.shape[0]):
            segmentation = labels[sample, 0]
            distance_foreground = dtedt(segmentation == 0)
            expanded_mask = (
                distance_foreground < inference_config.grow_distance
            )
            distance_background = dtedt(expanded_mask)
            segmentation[
                distance_background < inference_config.shrink_distance
            ] = 0
            pp_labels[sample, 0, ...] = segmentation
        return (
            prediction_layers
            + [(foreground, {"name": "Foreground"}, "labels")]
            + [(labels, {"name": "Segmentation"}, "labels")]
            + [(pp_labels, {"name": "Post Processed"}, "labels")]
        )

    @thread_worker
    def segment(self):
        threshold = self.binary_threshold_slider.value()
        print(f"Threshold is {threshold}")

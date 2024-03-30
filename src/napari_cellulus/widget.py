from pathlib import Path

import gunpowder as gp
import numpy as np
import pyqtgraph as pg
import torch
from attrs import asdict
from cellulus.configs.experiment_config import ExperimentConfig
from cellulus.configs.inference_config import InferenceConfig
from cellulus.configs.model_config import ModelConfig
from cellulus.configs.train_config import TrainConfig
from cellulus.criterions import get_loss
from cellulus.models import get_model
from cellulus.train import train_iteration
from cellulus.utils.mean_shift import mean_shift_segmentation
from cellulus.utils.misc import size_filter
from napari.qt.threading import thread_worker
from napari.utils.events import Event
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
)
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import distance_transform_edt as dtedt
from skimage.filters import threshold_otsu
from tqdm import tqdm

from .datasets.napari_dataset import NapariDataset
from .datasets.napari_image_source import NapariImageSource


class Model(torch.nn.Module):
    def __init__(self, model, selected_axes):
        super().__init__()
        self.model = model
        self.selected_axes = selected_axes

    def forward(self, raw):
        if "s" in self.selected_axes and "c" in self.selected_axes:
            pass
        elif "s" in self.selected_axes and "c" not in self.selected_axes:

            raw = torch.unsqueeze(raw, 1)
        elif "s" not in self.selected_axes and "c" in self.selected_axes:
            pass
        elif "s" not in self.selected_axes and "c" not in self.selected_axes:
            raw = torch.unsqueeze(raw, 1)
        return self.model(raw)

    @staticmethod
    def select_and_add_coordinates(outputs, coordinates):
        selections = []
        # outputs.shape = (b, c, h, w) or (b, c, d, h, w)
        for output, coordinate in zip(outputs, coordinates):
            if output.ndim == 3:
                selection = output[:, coordinate[:, 1], coordinate[:, 0]]
            elif output.ndim == 4:
                selection = output[
                    :, coordinate[:, 2], coordinate[:, 1], coordinate[:, 0]
                ]
            selection = selection.transpose(1, 0)
            selection += coordinate
            selections.append(selection)

        # selection.shape = (b, c, p) where p is the number of selected positions
        return torch.stack(selections, dim=0)

    def set_infer(self, p_salt_pepper, num_infer_iterations, device):
        self.model.eval()
        self.model.set_infer(p_salt_pepper, num_infer_iterations, device)


class Widget(QMainWindow):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.scroll = QScrollArea()
        # initialize outer layout
        layout = QVBoxLayout()

        # initialize individual grid layouts from top to bottom
        self.grid_0 = QGridLayout()  # title
        self.set_grid_0()
        self.grid_1 = QGridLayout()  # device
        self.set_grid_1()
        self.grid_2 = QGridLayout()  # raw image selector
        self.set_grid_2()
        self.grid_3 = QGridLayout()  # train configs
        self.set_grid_3()
        self.grid_4 = QGridLayout()  # model configs
        self.set_grid_4()
        self.grid_5 = QGridLayout()  # loss plot and train/stop button
        self.set_grid_5()
        self.grid_6 = QGridLayout()  # inference
        self.set_grid_6()
        self.grid_7 = QGridLayout()  # feedback
        self.set_grid_7()

        layout.addLayout(self.grid_0)
        layout.addLayout(self.grid_1)
        layout.addLayout(self.grid_2)
        layout.addLayout(self.grid_3)
        layout.addLayout(self.grid_4)
        layout.addLayout(self.grid_5)
        layout.addLayout(self.grid_6)
        layout.addLayout(self.grid_7)
        self.set_scroll_area(layout)
        self.viewer.layers.events.inserted.connect(self.update_raw_selector)
        self.viewer.layers.events.removed.connect(self.update_raw_selector)

    def update_raw_selector(self, event):
        count = 0
        for i in range(self.raw_selector.count() - 1, -1, -1):
            if self.raw_selector.itemText(i) == f"{event.value}":
                # remove item
                self.raw_selector.removeItem(i)
                count = 1
        if count == 0:
            self.raw_selector.addItems([f"{event.value}"])

    def set_grid_0(self):
        text_label = QLabel("<h3>Cellulus</h3>")
        method_description_label = QLabel(
            '<small>Unsupervised Learning of Object-Centric Embeddings<br>for Cell Instance Segmentation in Microscopy Images.<br>If you are using this in your research, please <a href="https://github.com/funkelab/cellulus#citation" style="color:gray;">cite us</a>.</small><br><small><tt><a href="https://github.com/funkelab/cellulus" style="color:gray;">https://github.com/funkelab/cellulus</a></tt></small>'
        )
        self.grid_0.addWidget(text_label, 0, 0, 1, 1)
        self.grid_0.addWidget(method_description_label, 1, 0, 2, 1)

    def set_grid_1(self):
        device_label = QLabel(self)
        device_label.setText("Device")
        self.device_combo_box = QComboBox(self)
        self.device_combo_box.addItem("cpu")
        self.device_combo_box.addItem("cuda:0")
        self.device_combo_box.addItem("mps")
        self.device_combo_box.setCurrentText("mps")
        self.grid_1.addWidget(device_label, 0, 0, 1, 1)
        self.grid_1.addWidget(self.device_combo_box, 0, 1, 1, 1)

    def set_grid_2(self):
        self.raw_selector = QComboBox(self)
        for layer in self.viewer.layers:
            self.raw_selector.addItem(f"{layer}")
        self.grid_2.addWidget(self.raw_selector, 0, 0, 1, 5)
        # Initialize Checkboxes
        self.s_check_box = QCheckBox("s/t")
        self.c_check_box = QCheckBox("c")
        self.z_check_box = QCheckBox("z")
        self.y_check_box = QCheckBox("y")
        self.x_check_box = QCheckBox("x")
        self.grid_2.addWidget(self.s_check_box, 1, 0, 1, 1)
        self.grid_2.addWidget(self.c_check_box, 1, 1, 1, 1)
        self.grid_2.addWidget(self.z_check_box, 1, 2, 1, 1)
        self.grid_2.addWidget(self.y_check_box, 1, 3, 1, 1)
        self.grid_2.addWidget(self.x_check_box, 1, 4, 1, 1)

    def set_grid_3(self):
        normalization_factor_label = QLabel(self)
        normalization_factor_label.setText("Normalization Factor")
        self.normalization_factor_line = QLineEdit(self)
        self.normalization_factor_line.setAlignment(Qt.AlignCenter)
        self.normalization_factor_line.setText("1.0")
        crop_size_label = QLabel(self)
        crop_size_label.setText("Crop Size")
        self.crop_size_line = QLineEdit(self)
        self.crop_size_line.setAlignment(Qt.AlignCenter)
        self.crop_size_line.setText("252")
        batch_size_label = QLabel(self)
        batch_size_label.setText("Batch Size")
        self.batch_size_line = QLineEdit(self)
        self.batch_size_line.setAlignment(Qt.AlignCenter)
        self.batch_size_line.setText("8")
        max_iterations_label = QLabel(self)
        max_iterations_label.setText("Max iterations")
        self.max_iterations_line = QLineEdit(self)
        self.max_iterations_line.setAlignment(Qt.AlignCenter)
        self.max_iterations_line.setText("5000")
        self.grid_3.addWidget(normalization_factor_label, 0, 0, 1, 1)
        self.grid_3.addWidget(self.normalization_factor_line, 0, 1, 1, 1)
        self.grid_3.addWidget(crop_size_label, 1, 0, 1, 1)
        self.grid_3.addWidget(self.crop_size_line, 1, 1, 1, 1)
        self.grid_3.addWidget(batch_size_label, 2, 0, 1, 1)
        self.grid_3.addWidget(self.batch_size_line, 2, 1, 1, 1)
        self.grid_3.addWidget(max_iterations_label, 3, 0, 1, 1)
        self.grid_3.addWidget(self.max_iterations_line, 3, 1, 1, 1)

    def set_grid_4(self):
        feature_maps_label = QLabel(self)
        feature_maps_label.setText("Number of feature maps")
        self.feature_maps_line = QLineEdit(self)
        self.feature_maps_line.setAlignment(Qt.AlignCenter)
        self.feature_maps_line.setText("24")
        feature_maps_increase_label = QLabel(self)
        feature_maps_increase_label.setText("Feature maps inc. factor")
        self.feature_maps_increase_line = QLineEdit(self)
        self.feature_maps_increase_line.setAlignment(Qt.AlignCenter)
        self.feature_maps_increase_line.setText("3")
        self.train_model_from_scratch_checkbox = QCheckBox(
            "Train from scratch"
        )
        self.train_model_from_scratch_checkbox.stateChanged.connect(
            self.effect_load_weights
        )
        self.load_model_button = QPushButton("Load weights")
        self.load_model_button.clicked.connect(self.load_weights)
        self.train_model_from_scratch_checkbox.setChecked(False)
        self.grid_4.addWidget(feature_maps_label, 0, 0, 1, 1)
        self.grid_4.addWidget(self.feature_maps_line, 0, 1, 1, 1)
        self.grid_4.addWidget(feature_maps_increase_label, 1, 0, 1, 1)
        self.grid_4.addWidget(self.feature_maps_increase_line, 1, 1, 1, 1)
        self.grid_4.addWidget(
            self.train_model_from_scratch_checkbox, 2, 0, 1, 1
        )
        self.grid_4.addWidget(self.load_model_button, 2, 1, 1, 1)

    def set_grid_5(self):
        self.losses_widget = pg.PlotWidget()
        self.losses_widget.setBackground((37, 41, 49))
        styles = {"color": "white", "font-size": "16px"}
        self.losses_widget.setLabel("left", "Loss", **styles)
        self.losses_widget.setLabel("bottom", "Iterations", **styles)
        self.start_training_button = QPushButton("Start training")
        self.start_training_button.setFixedSize(88, 30)
        self.stop_training_button = QPushButton("Stop training")
        self.stop_training_button.setFixedSize(88, 30)
        self.save_weights_button = QPushButton("Save weights")
        self.save_weights_button.setFixedSize(88, 30)

        self.grid_5.addWidget(self.losses_widget, 0, 0, 4, 4)
        self.grid_5.addWidget(self.start_training_button, 5, 0, 1, 1)
        self.grid_5.addWidget(self.stop_training_button, 5, 1, 1, 1)
        self.grid_5.addWidget(self.save_weights_button, 5, 2, 1, 1)

        self.start_training_button.clicked.connect(
            self.prepare_for_start_training
        )
        self.stop_training_button.clicked.connect(
            self.prepare_for_stop_training
        )
        self.save_weights_button.clicked.connect(self.save_weights)

    def set_grid_6(self):
        threshold_label = QLabel("Threshold")
        self.threshold_line = QLineEdit(self)
        self.threshold_line.textChanged.connect(self.prepare_thresholds)
        self.threshold_line.setAlignment(Qt.AlignCenter)
        self.threshold_line.setText(None)

        bandwidth_label = QLabel("Bandwidth")
        self.bandwidth_line = QLineEdit(self)
        self.bandwidth_line.setAlignment(Qt.AlignCenter)
        self.bandwidth_line.textChanged.connect(self.prepare_bandwidths)

        self.radio_button_group = QButtonGroup(self)
        self.radio_button_cell = QRadioButton("Cell")
        self.radio_button_nucleus = QRadioButton("Nucleus")
        self.radio_button_group.addButton(self.radio_button_nucleus)
        self.radio_button_group.addButton(self.radio_button_cell)

        self.radio_button_cell.setChecked(True)
        self.min_size_label = QLabel("Minimum Size")
        self.min_size_line = QLineEdit(self)
        self.min_size_line.setAlignment(Qt.AlignCenter)
        self.min_size_line.textChanged.connect(self.prepare_min_sizes)

        self.start_inference_button = QPushButton("Start inference")
        self.start_inference_button.setFixedSize(140, 30)
        self.stop_inference_button = QPushButton("Stop inference")
        self.stop_inference_button.setFixedSize(140, 30)

        self.grid_6.addWidget(threshold_label, 0, 0, 1, 1)
        self.grid_6.addWidget(self.threshold_line, 0, 1, 1, 1)

        self.grid_6.addWidget(bandwidth_label, 1, 0, 1, 1)
        self.grid_6.addWidget(self.bandwidth_line, 1, 1, 1, 1)
        self.grid_6.addWidget(self.radio_button_cell, 2, 0, 1, 1)
        self.grid_6.addWidget(self.radio_button_nucleus, 2, 1, 1, 1)
        self.grid_6.addWidget(self.min_size_label, 3, 0, 1, 1)
        self.grid_6.addWidget(self.min_size_line, 3, 1, 1, 1)
        self.grid_6.addWidget(self.start_inference_button, 4, 0, 1, 1)
        self.grid_6.addWidget(self.stop_inference_button, 4, 1, 1, 1)
        self.start_inference_button.clicked.connect(
            self.prepare_for_start_inference
        )
        self.stop_inference_button.clicked.connect(
            self.prepare_for_stop_inference
        )

    def set_grid_7(self):
        # Initialize Feedback Button
        feedback_label = QLabel(
            '<small>Please share any feedback <a href="https://github.com/funkelab/napari-cellulus/issues/new/choose" style="color:gray;">here</a>.</small>'
        )
        self.grid_7.addWidget(feedback_label, 0, 0, 2, 1)

    def set_scroll_area(self, layout):
        self.scroll.setLayout(layout)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)

        self.setFixedWidth(300)
        self.setCentralWidget(self.scroll)

    def get_selected_axes(self):
        names = []
        for name, check_box in zip(
            "sczyx",
            [
                self.s_check_box,
                self.c_check_box,
                self.z_check_box,
                self.y_check_box,
                self.x_check_box,
            ],
        ):
            if check_box.isChecked():
                names.append(name)

        return names

    def create_configs(self):
        if not hasattr(self, "train_config"):
            self.train_config = TrainConfig(
                crop_size=[int(self.crop_size_line.text())],
                batch_size=int(self.batch_size_line.text()),
                max_iterations=int(self.max_iterations_line.text()),
                device=self.device_combo_box.currentText(),
            )
        if not hasattr(self, "model_config"):
            self.model_config = ModelConfig(
                num_fmaps=int(self.feature_maps_line.text()),
                fmap_inc_factor=int(self.feature_maps_increase_line.text()),
            )
        if not hasattr(self, "experiment_config"):
            self.experiment_config = ExperimentConfig(
                train_config=asdict(self.train_config),
                model_config=asdict(self.model_config),
                normalization_factor=float(
                    self.normalization_factor_line.text()
                ),
            )
        if not hasattr(self, "losses"):
            self.losses = []
        if not hasattr(self, "iterations"):
            self.iterations = []
        if not hasattr(self, "start_iteration"):
            self.start_iteration = 0

        self.model_dir = "/tmp/models"
        self.threshold_line.setEnabled(False)
        self.bandwidth_line.setEnabled(False)
        self.min_size_line.setEnabled(False)

    def update_inference_widgets(self, event: Event):
        if self.s_check_box.isChecked():
            shape = event.value
            sample_index = shape[0]
            if (
                hasattr(self, "thresholds")
                and self.thresholds[sample_index] is not None
            ):
                self.threshold_line.setText(
                    str(round(self.thresholds[sample_index], 3))
                )
            if (
                hasattr(self, "band_widths")
                and self.band_widths[sample_index] is not None
            ):
                self.bandwidth_line.setText(
                    str(round(self.band_widths[sample_index], 3))
                )
            if (
                hasattr(self, "min_sizes")
                and self.min_sizes[sample_index] is not None
            ):
                self.min_size_line.setText(
                    str(round(self.min_sizes[sample_index], 3))
                )

    def prepare_for_start_training(self):
        self.start_training_button.setEnabled(False)
        self.stop_training_button.setEnabled(True)
        self.save_weights_button.setEnabled(False)
        self.threshold_line.setEnabled(False)
        self.bandwidth_line.setEnabled(False)
        self.radio_button_nucleus.setEnabled(False)
        self.radio_button_cell.setEnabled(False)
        self.min_size_line.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(False)

        self.train_worker = self.train()
        self.train_worker.yielded.connect(self.on_yield_training)
        self.train_worker.returned.connect(self.prepare_for_stop_training)
        self.train_worker.start()

    def remove_inference_attributes(self):
        if hasattr(self, "embeddings"):
            delattr(self, "embeddings")
        if hasattr(self, "detection"):
            delattr(self, "detection")
        if hasattr(self, "segmentation"):
            delattr(self, "segmentation")
        if hasattr(self, "thresholds"):
            delattr(self, "thresholds")
        if hasattr(self, "thresholds_last"):
            delattr(self, "thresholds_last")
        if hasattr(self, "band_widths"):
            delattr(self, "band_widths")
        if hasattr(self, "band_widths_last"):
            delattr(self, "band_widths_last")
        if hasattr(self, "min_sizes"):
            delattr(self, "min_sizes")
        if hasattr(self, "min_sizes_last"):
            delattr(self, "min_sizes_last")

    @thread_worker
    def train(self):
        self.create_configs()  # configs
        self.remove_inference_attributes()
        self.viewer.dims.events.current_step.connect(
            self.update_inference_widgets
        )  # listen to viewer slider

        for layer in self.viewer.layers:
            if f"{layer}" == self.raw_selector.currentText():
                raw_image_layer = layer
                break

        if not Path(self.model_dir).exists():
            Path(self.model_dir).mkdir()

        # Turn layer into dataset
        self.napari_dataset = NapariDataset(
            layer=raw_image_layer,
            axis_names=self.get_selected_axes(),
            crop_size=self.train_config.crop_size[0],  # list to integer
            density=self.train_config.density,
            kappa=self.train_config.kappa,
            normalization_factor=self.experiment_config.normalization_factor,
        )
        # Create dataloader
        train_dataloader = torch.utils.data.DataLoader(
            dataset=self.napari_dataset,
            batch_size=self.train_config.batch_size,
            drop_last=True,
            num_workers=self.train_config.num_workers,
            pin_memory=True,
        )
        # Set model
        model_original = get_model(
            in_channels=self.napari_dataset.get_num_channels()
            if self.napari_dataset.get_num_channels() != 0
            else 1,
            out_channels=self.napari_dataset.get_num_spatial_dims(),
            num_fmaps=self.model_config.num_fmaps,
            fmap_inc_factor=self.model_config.fmap_inc_factor,
            features_in_last_layer=self.model_config.features_in_last_layer,
            downsampling_factors=[
                tuple(factor)
                for factor in self.model_config.downsampling_factors
            ],
            num_spatial_dims=self.napari_dataset.get_num_spatial_dims(),
        )

        # Set device
        self.device = torch.device(self.train_config.device)

        model = Model(
            model=model_original, selected_axes=self.get_selected_axes()
        )
        self.model = model.to(self.device)

        # Initialize model weights
        if self.model_config.initialize:
            for _name, layer in self.model.named_modules():
                if isinstance(layer, torch.nn.modules.conv._ConvNd):
                    torch.nn.init.kaiming_normal_(
                        layer.weight, nonlinearity="relu"
                    )

        # Set loss
        criterion = get_loss(
            regularizer_weight=self.train_config.regularizer_weight,
            temperature=self.train_config.temperature,
            density=self.train_config.density,
            num_spatial_dims=self.napari_dataset.get_num_spatial_dims(),
            device=self.device,
        )

        # Set optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config.initial_learning_rate,
            weight_decay=0.01,
        )
        if hasattr(self, "pre_trained_model_checkpoint"):
            self.model_config.checkpoint = self.pre_trained_model_checkpoint

        # Resume training
        if self.train_model_from_scratch_checkbox.isChecked():
            self.losses, self.iterations = [], []
            self.start_iteration = 0
            self.losses_widget.clear()

        else:
            if self.model_config.checkpoint is None:
                pass
            else:
                print(f"Resuming model from {self.model_config.checkpoint}")
                state = torch.load(
                    self.model_config.checkpoint, map_location=self.device
                )
                self.start_iteration = state["iterations"][-1] + 1
                self.model.load_state_dict(
                    state["model_state_dict"], strict=True
                )
                self.optimizer.load_state_dict(state["optim_state_dict"])
                self.losses, self.iterations = (
                    state["losses"],
                    state["iterations"],
                )

        # Call Train Iteration

        for iteration, batch in tqdm(
            zip(
                range(self.start_iteration, self.train_config.max_iterations),
                train_dataloader,
            )
        ):
            loss, oce_loss, prediction = train_iteration(
                batch,
                model=self.model,
                criterion=criterion,
                optimizer=self.optimizer,
                device=self.device,
            )
            yield loss, iteration
        return

    def on_yield_training(self, loss_iteration):
        loss, iteration = loss_iteration
        print(f"===> Iteration: {iteration}, loss: {loss:.6f}")
        self.iterations.append(iteration)
        self.losses.append(loss)
        self.losses_widget.plot(self.iterations, self.losses)

    def prepare_for_stop_training(self):
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(True)
        self.save_weights_button.setEnabled(True)
        if not hasattr(self, "thresholds"):
            self.threshold_line.setEnabled(False)
        else:
            self.threshold_line.setEnabled(True)
        if not hasattr(self, "band_widths"):
            self.bandwidth_line.setEnabled(False)
        else:
            self.bandwidth_line.setEnabled(True)
        self.radio_button_nucleus.setEnabled(True)
        self.radio_button_cell.setEnabled(True)
        if not hasattr(self, "min_sizes"):
            self.min_size_line.setEnabled(False)
        else:
            self.min_size_line.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(True)
        if self.train_worker is not None:
            state = {
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "iterations": self.iterations,
                "losses": self.losses,
            }
            checkpoint_file_name = Path("/tmp/models") / "last.pth"
            torch.save(state, checkpoint_file_name)
            self.train_worker.quit()
            self.model_config.checkpoint = checkpoint_file_name

    def prepare_for_start_inference(self):

        self.start_training_button.setEnabled(False)
        self.stop_training_button.setEnabled(False)
        self.save_weights_button.setEnabled(False)
        self.threshold_line.setEnabled(False)
        self.bandwidth_line.setEnabled(False)
        self.radio_button_nucleus.setEnabled(False)
        self.radio_button_cell.setEnabled(False)
        self.min_size_line.setEnabled(False)
        self.start_inference_button.setEnabled(False)
        self.stop_inference_button.setEnabled(True)

        self.inference_config = InferenceConfig(
            crop_size=[min(self.napari_dataset.get_spatial_array()) + 16],
            post_processing="cell"
            if self.radio_button_cell.isChecked()
            else "nucleus",
        )

        self.inference_worker = self.infer()
        self.inference_worker.returned.connect(self.on_return_infer)
        self.inference_worker.start()

    def prepare_for_stop_inference(self):
        self.start_training_button.setEnabled(True)
        self.stop_training_button.setEnabled(True)
        self.save_weights_button.setEnabled(True)

        self.threshold_line.setEnabled(True)
        self.bandwidth_line.setEnabled(True)
        self.radio_button_nucleus.setEnabled(True)
        self.radio_button_cell.setEnabled(True)
        self.min_size_line.setEnabled(True)
        self.start_inference_button.setEnabled(True)
        self.stop_inference_button.setEnabled(True)
        if self.napari_dataset.get_num_samples() == 0:
            self.threshold_line.setText(str(round(self.thresholds[0], 3)))
            self.bandwidth_line.setText(str(round(self.band_widths[0], 3)))
            self.min_size_line.setText(str(round(self.min_sizes[0], 3)))
        if self.inference_worker is not None:
            self.inference_worker.quit()

    @thread_worker
    def infer(self):
        for layer in self.viewer.layers:
            if f"{layer}" == self.raw_selector.currentText():
                raw_image_layer = layer
                break

        if not hasattr(self, "thresholds"):
            self.thresholds = (
                [None] * self.napari_dataset.get_num_samples()
                if self.napari_dataset.get_num_samples() != 0
                else [None] * 1
            )

        if not hasattr(self, "thresholds_last"):
            self.thresholds_last = self.thresholds.copy()

        if not hasattr(self, "band_widths") and (
            self.inference_config.bandwidth is None
        ):
            self.band_widths = (
                [0.25 * self.experiment_config.object_size]
                * self.napari_dataset.get_num_samples()
                if self.napari_dataset.get_num_samples() != 0
                else [0.25 * self.experiment_config.object_size]
            )

        if not hasattr(self, "band_widths_last"):
            self.band_widths_last = self.band_widths.copy()

        if (
            not hasattr(self, "min_sizes")
            and self.inference_config.min_size is None
        ):
            if self.napari_dataset.get_num_spatial_dims() == 2:
                self.min_sizes = (
                    [
                        int(
                            0.1
                            * np.pi
                            * (self.experiment_config.object_size**2)
                            / 4
                        )
                    ]
                    * self.napari_dataset.get_num_samples()
                    if self.napari_dataset.get_num_samples() != 0
                    else [
                        int(
                            0.1
                            * np.pi
                            * (self.experiment_config.object_size**2)
                            / 4
                        )
                    ]
                )
            elif (
                self.napari_dataset.get_num_spatial_dims() == 3
                and len(self.min_sizes) == 0
            ):
                self.min_sizes = (
                    [
                        int(
                            0.1
                            * 4.0
                            / 3.0
                            * np.pi
                            * (self.experiment_config.object_size**3)
                            / 8
                        )
                    ]
                    * self.napari_dataset.get_num_samples()
                    if self.napari_dataset.get_num_samples() != 0
                    else [
                        int(
                            0.1
                            * 4.0
                            / 3.0
                            * np.pi
                            * (self.experiment_config.object_size**3)
                            / 8
                        )
                    ]
                )

        if not hasattr(self, "min_sizes_last"):
            self.min_sizes_last = self.min_sizes.copy()

        # set in eval mode
        self.model = self.model.to(self.device)

        self.model.eval()
        self.model.set_infer(
            p_salt_pepper=self.inference_config.p_salt_pepper,
            num_infer_iterations=self.inference_config.num_infer_iterations,
            device=self.device,
        )

        if self.napari_dataset.get_num_spatial_dims() == 2:
            crop_size_tuple = (self.inference_config.crop_size[0],) * 2
            predicted_crop_size_tuple = (
                self.inference_config.crop_size[0] - 16,
            ) * 2
        elif self.napari_dataset.get_num_spatial_dims() == 3:
            crop_size_tuple = (self.inference_config.crop_size[0],) * 3
            predicted_crop_size_tuple = (
                self.inference_config.crop_size[0] - 16,
            ) * 3

        input_shape = gp.Coordinate(
            (
                1,
                self.napari_dataset.get_num_channels()
                if self.napari_dataset.get_num_channels() != 0
                else 1,
                *crop_size_tuple,
            )
        )

        if self.napari_dataset.get_num_channels() == 0:
            output_shape = gp.Coordinate(
                self.model(
                    torch.zeros((1, *crop_size_tuple), dtype=torch.float32).to(
                        self.device
                    )
                ).shape
            )
        else:
            output_shape = gp.Coordinate(
                self.model(
                    torch.zeros(
                        (
                            1,
                            self.napari_dataset.get_num_channels(),
                            *crop_size_tuple,
                        ),
                        dtype=torch.float32,
                    ).to(self.device)
                ).shape
            )

        voxel_size = (
            (1,) * 2
            if self.napari_dataset.get_num_spatial_dims() == 2
            else (1,) * 3
        )

        input_size = gp.Coordinate(input_shape[2:]) * gp.Coordinate(voxel_size)
        output_size = gp.Coordinate(output_shape[2:]) * gp.Coordinate(
            voxel_size
        )
        context = (input_size - output_size) // 2
        raw = gp.ArrayKey("RAW")
        prediction = gp.ArrayKey("PREDICT")
        scan_request = gp.BatchRequest()

        # scan_request.add(raw, input_size)
        scan_request[raw] = gp.Roi(
            (-8,) * self.napari_dataset.get_num_spatial_dims(),
            crop_size_tuple,
        )
        scan_request[prediction] = gp.Roi(
            (0,) * self.napari_dataset.get_num_spatial_dims(),
            predicted_crop_size_tuple,
        )

        predict = gp.torch.Predict(
            self.model,
            inputs={"raw": raw},
            outputs={0: prediction},
            array_specs={prediction: gp.ArraySpec(voxel_size=voxel_size)},
        )

        pipeline = NapariImageSource(
            image=raw_image_layer,
            key=raw,
            spec=gp.ArraySpec(
                gp.Roi(
                    (0,) * self.napari_dataset.get_num_spatial_dims(),
                    raw_image_layer.data.shape[
                        -self.napari_dataset.get_num_spatial_dims() :
                    ],
                ),
                voxel_size=(1,) * self.napari_dataset.get_num_spatial_dims(),
            ),
            spatial_dims=self.napari_dataset.get_spatial_dims(),
        )

        if self.napari_dataset.get_num_samples() == 0:
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

        request = gp.BatchRequest()
        request.add(
            prediction,
            raw_image_layer.data.shape[
                -self.napari_dataset.get_num_spatial_dims() :
            ],
        )

        # Obtain Embeddings
        print("Predicting Embeddings ...")

        if hasattr(self, "embeddings"):
            pass
        else:
            with gp.build(pipeline):
                batch = pipeline.request_batch(request)

            self.embeddings = batch.arrays[prediction].data
        embeddings_centered = np.zeros_like(self.embeddings)
        foreground_mask = np.zeros_like(
            self.embeddings[:, 0:1, ...], dtype=bool
        )
        colormaps = ["red", "green", "blue"]

        # Obtain Object Centered Embeddings
        for sample in tqdm(range(self.embeddings.shape[0])):
            embeddings_sample = self.embeddings[sample]
            embeddings_std = embeddings_sample[-1, ...]
            embeddings_mean = embeddings_sample[
                np.newaxis, : self.napari_dataset.get_num_spatial_dims(), ...
            ].copy()
            if self.thresholds[sample] is None:
                threshold = threshold_otsu(embeddings_std)
                self.thresholds[sample] = round(threshold, 3)
            binary_mask = embeddings_std < self.thresholds[sample]
            foreground_mask[sample] = binary_mask[np.newaxis, ...]
            embeddings_centered_sample = embeddings_sample.copy()
            embeddings_mean_masked = (
                binary_mask[np.newaxis, np.newaxis, ...] * embeddings_mean
            )
            if embeddings_centered_sample.shape[0] == 3:
                c_x = embeddings_mean_masked[0, 0]
                c_y = embeddings_mean_masked[0, 1]
                c_x = c_x[c_x != 0].mean()
                c_y = c_y[c_y != 0].mean()
                embeddings_centered_sample[0] -= c_x
                embeddings_centered_sample[1] -= c_y
            elif embeddings_centered_sample.shape[0] == 3:
                c_x = embeddings_mean_masked[0, 0]
                c_y = embeddings_mean_masked[0, 1]
                c_z = embeddings_mean_masked[0, 2]
                c_x = c_x[c_x != 0].mean()
                c_y = c_y[c_y != 0].mean()
                c_z = c_z[c_z != 0].mean()
                embeddings_centered_sample[0] -= c_x
                embeddings_centered_sample[1] -= c_y
                embeddings_centered_sample[2] -= c_z

            embeddings_centered[sample] = embeddings_centered_sample

        embeddings_layers = [
            (
                embeddings_centered[:, i : i + 1, ...].copy(),
                {
                    "name": "Offset ("
                    + "zyx"[self.napari_dataset.get_num_spatial_dims() - i]
                    + ")"
                    if i < self.napari_dataset.get_num_spatial_dims()
                    else "Uncertainty",
                    "colormap": colormaps[
                        self.napari_dataset.get_num_spatial_dims() - i
                    ]
                    if i < self.napari_dataset.get_num_spatial_dims()
                    else "gray",
                    "blending": "additive",
                },
                "image",
            )
            for i in range(self.napari_dataset.get_num_spatial_dims() + 1)
        ]

        print("Clustering Objects in the obtained Foreground Mask ...")
        if hasattr(self, "detection"):
            pass
        else:
            self.detection = np.zeros_like(
                self.embeddings[:, 0:1, ...], dtype=np.uint16
            )
        for sample in tqdm(range(self.embeddings.shape[0])):
            embeddings_sample = self.embeddings[sample]
            embeddings_std = embeddings_sample[-1, ...]
            embeddings_mean = embeddings_sample[
                np.newaxis, : self.napari_dataset.get_num_spatial_dims(), ...
            ].copy()

            if (
                self.thresholds[sample] != self.thresholds_last[sample]
                or self.band_widths[sample] != self.band_widths_last[sample]
            ):
                detection_sample = mean_shift_segmentation(
                    embeddings_mean,
                    embeddings_std,
                    bandwidth=self.band_widths[sample],
                    min_size=self.inference_config.min_size,
                    reduction_probability=self.inference_config.reduction_probability,
                    threshold=self.thresholds[sample],
                    seeds=None,
                )
                self.detection[sample, 0, ...] = detection_sample
            self.thresholds_last[sample] = self.thresholds[sample]
            self.band_widths_last[sample] = self.band_widths[sample]

        print("Converting Detections to Segmentations ...")
        if hasattr(self, "segmentation"):
            pass
        else:
            self.segmentation = np.zeros_like(
                self.embeddings[:, 0:1, ...], dtype=np.uint16
            )
        if self.radio_button_cell.isChecked():
            for sample in tqdm(range(self.embeddings.shape[0])):
                segmentation_sample = self.detection[sample, 0].copy()
                distance_foreground = dtedt(segmentation_sample == 0)
                expanded_mask = (
                    distance_foreground < self.inference_config.grow_distance
                )
                distance_background = dtedt(expanded_mask)
                segmentation_sample[
                    distance_background < self.inference_config.shrink_distance
                ] = 0
                self.segmentation[sample, 0, ...] = segmentation_sample
        elif self.radio_button_nucleus.isChecked():
            raw_image = raw_image_layer.data
            for sample in tqdm(range(self.embeddings.shape[0])):
                segmentation_sample = self.detection[sample, 0]
                if (
                    self.napari_dataset.get_num_samples() == 0
                    and self.napari_dataset.get_num_channels() == 0
                ):
                    raw_image_sample = raw_image
                elif (
                    self.napari_dataset.get_num_samples() != 0
                    and self.napari_dataset.get_num_channels() == 0
                ):
                    raw_image_sample = raw_image[sample]
                elif (
                    self.napari_dataset.get_num_samples() == 0
                    and self.napari_dataset.get_num_channels() != 0
                ):
                    raw_image_sample = raw_image[0]
                else:
                    raw_image_sample = raw_image[sample, 0]

                ids = np.unique(segmentation_sample)
                ids = ids[ids != 0]

                for id_ in ids:
                    segmentation_id_mask = segmentation_sample == id_
                    if self.napari_dataset.get_num_spatial_dims() == 2:
                        y, x = np.where(segmentation_id_mask)
                        y_min, y_max, x_min, x_max = (
                            np.min(y),
                            np.max(y),
                            np.min(x),
                            np.max(x),
                        )
                    elif self.napari_dataset.get_num_spatial_dims() == 3:
                        z, y, x = np.where(segmentation_id_mask)
                        z_min, z_max, y_min, y_max, x_min, x_max = (
                            np.min(z),
                            np.max(z),
                            np.min(y),
                            np.max(y),
                            np.min(x),
                            np.max(x),
                        )
                    raw_image_masked = raw_image_sample[segmentation_id_mask]
                    threshold = threshold_otsu(raw_image_masked)
                    mask = segmentation_id_mask & (
                        raw_image_sample > threshold
                    )

                    if self.napari_dataset.get_num_spatial_dims() == 2:
                        mask_small = binary_fill_holes(
                            mask[y_min : y_max + 1, x_min : x_max + 1]
                        )
                        mask[y_min : y_max + 1, x_min : x_max + 1] = mask_small
                        y, x = np.where(mask)
                        self.segmentation[sample, 0, y, x] = id_
                    elif self.napari_dataset.get_num_spatial_dims() == 3:
                        mask_small = binary_fill_holes(
                            mask[
                                z_min : z_max + 1,
                                y_min : y_max + 1,
                                x_min : x_max + 1,
                            ]
                        )
                        mask[
                            z_min : z_max + 1,
                            y_min : y_max + 1,
                            x_min : x_max + 1,
                        ] = mask_small
                        z, y, x = np.where(mask)
                        self.segmentation[sample, 0, z, y, x] = id_

        print("Removing small objects ...")

        # size filter - remove small objects
        for sample in tqdm(range(self.embeddings.shape[0])):
            if self.min_sizes[sample] != self.min_sizes_last[sample]:
                self.segmentation[sample, 0, ...] = size_filter(
                    self.segmentation[sample, 0], self.min_sizes[sample]
                )
            self.min_sizes_last[sample] = self.min_sizes[sample]
        return (
            embeddings_layers
            + [(foreground_mask, {"name": "Foreground Mask"}, "labels")]
            + [(self.detection, {"name": "Detection"}, "labels")]
            + [(self.segmentation, {"name": "Segmentation"}, "labels")]
        )

    def on_return_infer(self, layers):

        if "Offset (x)" in self.viewer.layers:
            del self.viewer.layers["Offset (x)"]
        if "Offset (y)" in self.viewer.layers:
            del self.viewer.layers["Offset (y)"]
        if "Offset (z)" in self.viewer.layers:
            del self.viewer.layers["Offset (z)"]
        if "Uncertainty" in self.viewer.layers:
            del self.viewer.layers["Uncertainty"]
        if "Foreground Mask" in self.viewer.layers:
            del self.viewer.layers["Foreground Mask"]
        if "Segmentation" in self.viewer.layers:
            del self.viewer.layers["Segmentation"]
        if "Detection" in self.viewer.layers:
            del self.viewer.layers["Detection"]

        for data, metadata, layer_type in layers:
            if layer_type == "image":
                self.viewer.add_image(data, **metadata)
            elif layer_type == "labels":
                if (
                    self.napari_dataset.get_num_samples() != 0
                    and self.napari_dataset.get_num_channels() != 0
                ):
                    self.viewer.add_labels(data.astype(int), **metadata)
                else:
                    self.viewer.add_labels(data[:, 0].astype(int), **metadata)
        self.viewer.layers["Offset (x)"].visible = False
        self.viewer.layers["Offset (y)"].visible = False
        self.viewer.layers["Uncertainty"].visible = False
        self.viewer.layers["Foreground Mask"].visible = False
        self.viewer.layers["Detection"].visible = False
        self.viewer.layers["Segmentation"].visible = True
        self.inference_worker.quit()
        self.prepare_for_stop_inference()

    def prepare_thresholds(self):
        sample_index = self.viewer.dims.current_step[0]
        self.thresholds[sample_index] = float(self.threshold_line.text())

    def prepare_bandwidths(self):
        sample_index = self.viewer.dims.current_step[0]
        self.band_widths[sample_index] = float(self.bandwidth_line.text())

    def prepare_min_sizes(self):
        sample_index = self.viewer.dims.current_step[0]
        self.min_sizes[sample_index] = float(self.min_size_line.text())

    def load_weights(self):
        file_name, _ = QFileDialog.getOpenFileName(
            caption="Load Model Weights"
        )
        self.pre_trained_model_checkpoint = file_name
        print(
            f"Model weights will be loaded from {self.pre_trained_model_checkpoint}"
        )

    def effect_load_weights(self):
        if self.train_model_from_scratch_checkbox.isChecked():
            self.load_model_button.setEnabled(False)
        else:
            self.load_model_button.setEnabled(True)

    def save_weights(self):
        checkpoint_file_name, _ = QFileDialog.getSaveFileName(
            caption="Save Model Weights"
        )
        if (
            hasattr(self, "model")
            and hasattr(self, "optimizer")
            and hasattr(self, "iterations")
            and hasattr(self, "losses")
        ):
            state = {
                "model_state_dict": self.model.state_dict(),
                "optim_state_dict": self.optimizer.state_dict(),
                "iterations": self.iterations,
                "losses": self.losses,
            }
            torch.save(state, checkpoint_file_name)
            print(f"Model weights will be saved at {checkpoint_file_name}")

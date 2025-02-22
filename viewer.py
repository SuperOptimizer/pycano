import numpy as np
import zarr
from typing import List, Tuple
import napari
import skimage
from numba import njit, prange
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDoubleSpinBox,
    QCheckBox, QPushButton, QFrame
)
from qtpy.QtCore import Qt

import glcae


@njit
def get_neighbors_3d(z, y, x, depth, height, width):
    directions = np.array([
        [-1, 0, 0], [1, 0, 0],
        [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1]
    ])

    neighbors = []

    for i in range(6):
        nz = z + directions[i, 0]
        ny = y + directions[i, 1]
        nx = x + directions[i, 2]

        if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
            neighbors.append((nz, ny, nx))

    return neighbors


@njit
def flood_fill_f32(volume, iso_threshold, start_threshold):
    depth, height, width = volume.shape
    mask = np.zeros_like(volume, dtype=np.uint8)
    visited = np.zeros_like(volume, dtype=np.uint8)

    iso_threshold_u8 = min(255, max(0, int(iso_threshold)))
    start_threshold_u8 = min(255, max(0, int(start_threshold)))

    max_queue_size = depth * height * width
    queue_z = np.zeros(max_queue_size, dtype=np.int32)
    queue_y = np.zeros(max_queue_size, dtype=np.int32)
    queue_x = np.zeros(max_queue_size, dtype=np.int32)
    queue_start = 0
    queue_end = 0

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if volume[z, y, x] >= start_threshold_u8:
                    queue_z[queue_end] = z
                    queue_y[queue_end] = y
                    queue_x[queue_end] = x
                    queue_end += 1

                    mask[z, y, x] = 1
                    visited[z, y, x] = 1

    directions = np.array([
        [-1, 0, 0], [1, 0, 0],
        [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1]
    ])

    while queue_start < queue_end:
        current_z = queue_z[queue_start]
        current_y = queue_y[queue_start]
        current_x = queue_x[queue_start]
        queue_start += 1

        for i in range(6):
            nz = current_z + directions[i, 0]
            ny = current_y + directions[i, 1]
            nx = current_x + directions[i, 2]

            if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                if visited[nz, ny, nx] == 0 and volume[nz, ny, nx] >= iso_threshold_u8:
                    mask[nz, ny, nx] = 1
                    visited[nz, ny, nx] = 1

                    queue_z[queue_end] = nz
                    queue_y[queue_end] = ny
                    queue_x[queue_end] = nx
                    queue_end += 1

    return mask


@njit(parallel=True)
def segment_and_clean_u8(volume_u8, iso_threshold=127, start_threshold=200):
    mask = flood_fill_f32(volume_u8, iso_threshold, start_threshold)

    result = np.zeros_like(volume_u8)
    for z in prange(volume_u8.shape[0]):
        for y in range(volume_u8.shape[1]):
            for x in range(volume_u8.shape[2]):
                if mask[z, y, x]:
                    result[z, y, x] = volume_u8[z, y, x]
                else:
                    result[z, y, x] = 0

    return result


@njit(parallel=True)
def segment_and_clean_f32(volume_u8, iso_threshold=127, start_threshold=200):
    mask = flood_fill_f32(volume_u8, iso_threshold, start_threshold)

    result = np.zeros_like(volume_u8)
    for z in prange(volume_u8.shape[0]):
        for y in range(volume_u8.shape[1]):
            for x in range(volume_u8.shape[2]):
                if mask[z, y, x]:
                    result[z, y, x] = volume_u8[z, y, x]
                else:
                    result[z, y, x] = 0

    return result


@njit(parallel=True)
def avgpool_denoise_3d(volume_u8, kernel=3):
    depth, height, width = volume_u8.shape
    result = np.zeros_like(volume_u8)
    half = kernel // 2

    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                values = []

                for zi in range(-half, half + 1):
                    for yi in range(-half, half + 1):
                        for xi in range(-half, half + 1):
                            nz, ny, nx = z + zi, y + yi, x + xi

                            if not (0 <= nz < depth and 0 <= ny < height and 0 <= nx < width):
                                continue

                            values.append(volume_u8[nz, ny, nx])

                if values:
                    result[z, y, x] = int(np.mean(values))

    return result


@njit
def avgpool_denoise_3d_fast(volume_u8, kernel=3):
    depth, height, width = volume_u8.shape
    result = np.zeros_like(volume_u8)
    half = kernel // 2

    padded = np.zeros((depth + 2 * half, height + 2 * half, width + 2 * half), dtype=np.uint8)
    padded[half:half + depth, half:half + height, half:half + width] = volume_u8

    for z in prange(depth):
        for y in range(height):
            for x in range(width):
                neighborhood = padded[z:z + kernel, y:y + kernel, x:x + kernel]
                result[z, y, x] = int(np.mean(neighborhood))

    return result


class ChannelControls(QWidget):
    def __init__(self, name, layer, color, parent=None):
        super().__init__(parent)
        self.name = name
        self.layer = layer
        self.color = color

        layout = QVBoxLayout()
        self.setLayout(layout)

        header = QLabel(f"<h3 style='color:{color}'>{name}</h3>")
        layout.addWidget(header)

        # Attenuation control - greatly extended range
        att_layout = QHBoxLayout()
        att_layout.addWidget(QLabel("Attenuation:"))

        self.att_slider = QSlider(Qt.Horizontal)
        self.att_slider.setMinimum(1)  # 0.0001 minimum (1/10000)
        self.att_slider.setMaximum(100000)  # 10.0 maximum (100000/10000)
        self.att_slider.setValue(int(layer.attenuation * 10000))
        self.att_slider.valueChanged.connect(self.update_attenuation)
        att_layout.addWidget(self.att_slider)

        self.att_value = QDoubleSpinBox()
        self.att_value.setRange(0.0001, 10.0)
        self.att_value.setDecimals(4)  # Show 4 decimal places
        self.att_value.setValue(layer.attenuation)
        self.att_value.setSingleStep(0.01)
        self.att_value.valueChanged.connect(self.update_attenuation_from_spinbox)
        att_layout.addWidget(self.att_value)

        layout.addLayout(att_layout)

        # Gamma control - greatly extended range
        gamma_layout = QHBoxLayout()
        gamma_layout.addWidget(QLabel("Gamma:"))

        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setMinimum(1)  # 0.0001 minimum
        self.gamma_slider.setMaximum(100000)  # 10.0 maximum
        self.gamma_slider.setValue(int(layer.gamma * 10000))
        self.gamma_slider.valueChanged.connect(self.update_gamma)
        gamma_layout.addWidget(self.gamma_slider)

        self.gamma_value = QDoubleSpinBox()
        self.gamma_value.setRange(0.0001, 10.0)
        self.gamma_value.setDecimals(4)  # Show 4 decimal places
        self.gamma_value.setValue(layer.gamma)
        self.gamma_value.setSingleStep(0.01)
        self.gamma_value.valueChanged.connect(self.update_gamma_from_spinbox)
        gamma_layout.addWidget(self.gamma_value)

        layout.addLayout(gamma_layout)

        # Visibility toggle
        vis_layout = QHBoxLayout()
        vis_layout.addWidget(QLabel("Visible:"))
        self.vis_checkbox = QCheckBox()
        self.vis_checkbox.setChecked(layer.visible)
        self.vis_checkbox.stateChanged.connect(self.toggle_visibility)
        vis_layout.addWidget(self.vis_checkbox)

        layout.addLayout(vis_layout)

    def update_attenuation(self, value):
        att_value = value / 10000.0  # Convert from slider value to actual value
        self.layer.attenuation = att_value
        self.att_value.blockSignals(True)
        self.att_value.setValue(att_value)
        self.att_value.blockSignals(False)

    def update_attenuation_from_spinbox(self, value):
        self.layer.attenuation = value
        self.att_slider.blockSignals(True)
        self.att_slider.setValue(int(value * 10000))
        self.att_slider.blockSignals(False)

    def update_gamma(self, value):
        gamma_value = value / 10000.0  # Convert from slider value to actual value
        self.layer.gamma = gamma_value
        self.gamma_value.blockSignals(True)
        self.gamma_value.setValue(gamma_value)
        self.gamma_value.blockSignals(False)

    def update_gamma_from_spinbox(self, value):
        self.layer.gamma = value
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(int(value * 10000))
        self.gamma_slider.blockSignals(False)

    def toggle_visibility(self, state):
        self.layer.visible = bool(state)


class VolumeControlsDock(QWidget):
    def __init__(self, layers, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Volume Controls")

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.channel_controls = []
        colors = ['red', 'green', 'blue']
        names = ['Red Channel', 'Green Channel', 'Blue Channel']

        for i, (layer, color, name) in enumerate(zip(layers, colors, names)):
            controls = ChannelControls(name, layer, color)
            layout.addWidget(controls)
            self.channel_controls.append(controls)

            if i < len(layers) - 1:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Sunken)
                layout.addWidget(separator)

        global_layout = QVBoxLayout()
        global_label = QLabel("<h3>Global Settings</h3>")
        global_layout.addWidget(global_label)

        reset_button = QPushButton("Reset All")
        reset_button.clicked.connect(self.reset_all)
        global_layout.addWidget(reset_button)

        layout.addLayout(global_layout)
        layout.addStretch()

    def reset_all(self):
        for control in self.channel_controls:
            control.att_value.setValue(0.1)
            control.gamma_value.setValue(1.0)
            control.vis_checkbox.setChecked(True)


class SimpleRGBVolumeViewer:
    def __init__(self,
                 zarr_paths: List[str],
                 start_coords: Tuple[int, int, int] = (0, 0, 0),
                 size_coords: Tuple[int, int, int] = None):
        if len(zarr_paths) != 3:
            raise ValueError("Need exactly three zarr paths")

        arrays = [zarr.open(path, mode='r') for path in zarr_paths]
        print(f"Input data type: {arrays[0].dtype}")
        self.shape = arrays[0].shape
        self.start = start_coords

        if size_coords is None:
            self.size = self.shape
            self.size = self.size[0], self.size[1], self.size[2]
        else:
            self.size = size_coords

        self.channels = self._load_channels(arrays)
        self.viewer = napari.Viewer(ndisplay=3)
        self._add_color_channels()
        self._add_control_dock()

    def _load_channels(self, arrays):
        channels = []
        for i, arr in enumerate(arrays):
            chunk = arr[
                    self.start[0]:self.start[0] + self.size[0],
                    self.start[1]:self.start[1] + self.size[1],
                    self.start[2]:self.start[2] + self.size[2]
                    ]
            chunk[chunk < 32] = 1
            min_val = np.min(chunk)
            max_val = np.max(chunk)
            if max_val > min_val:
                chunk = (((chunk - min_val) / (max_val - min_val)) * 255).astype(np.uint8)
            else:
                raise ValueError("Zero range in channel")
            chunk = skimage.exposure.equalize_hist(chunk)
            chunk = glcae.global_and_local_contrast_enhancement_3d(chunk)
            chunk = avgpool_denoise_3d_fast(chunk,kernel=3)
            min_val = np.min(chunk)
            max_val = np.max(chunk)
            chunk = (((chunk - min_val) / (max_val - min_val)) * 255).astype(np.uint8)

            channels.append(chunk)
            print(
                f"Channel {i} stats after conversion: min={np.min(chunk)}, max={np.max(chunk)}, mean={np.mean(chunk):.2f}")
        return channels

        return [
            np.abs(channels[1] - channels[0]),
            np.abs(channels[1]),
            np.abs(channels[1] - channels[2])
        ]

        return [
            np.abs(channels[0] - channels[1]),
            np.abs(channels[1] - channels[2]),
            np.abs(channels[2] - channels[0])
        ]

    def _add_color_channels(self):
        colors = ['red', 'green', 'blue']
        names = ['Red Channel', 'Green Channel', 'Blue Channel']

        self.layers = []

        for i, (channel, color, name) in enumerate(zip(self.channels, colors, names)):
            layer = self.viewer.add_image(
                channel,
                name=name,
                colormap=color,
                blending='additive',
                rendering='attenuated_mip',
                attenuation=0.1,
                contrast_limits=[0, 255],
                gamma=1.0,
            )
            self.layers.append(layer)

        print(f"Added 3 separate channels with individual coloring")

    def _add_control_dock(self):
        self.control_dock = VolumeControlsDock(self.layers)
        self.viewer.window.add_dock_widget(
            self.control_dock,
            name="Volume Controls",
            area="right"
        )

    def run(self):
        print(f"Starting viewer with 3 colored channels and custom controls")
        napari.run()


if __name__ == "__main__":
    paths = [
        "/Users/forrest/frag6/53kev_3",
        "/Users/forrest/frag6/70kev_3",
        "/Users/forrest/frag6/88kev_3"
    ]

    viewer = SimpleRGBVolumeViewer(
        paths,
        (0, 0, 0),
        (512, 512, 512)
    )
    viewer.run()
import numpy as np
import zarr
from typing import List, Tuple
import napari
from magicgui import magicgui
from scipy import ndimage


class RGBVolumeViewer:
    def __init__(self,
                 zarr_paths: List[str],
                 start_coords: Tuple[int, int, int],
                 size_coords: Tuple[int, int, int],
                 threshold: int = 32,
                 sigma: float = 1.0):
        """
        Initialize viewer for RGB volume visualization.

        Args:
            zarr_paths: Paths to three zarr files [red, green, blue]
            start_coords: (z,y,x) starting coordinates for chunk extraction
            size_coords: (z,y,x) size of chunk to extract
            threshold: ISO threshold for visualization
            sigma: Gaussian blur sigma
        """
        if len(zarr_paths) != 3:
            raise ValueError("Need exactly three zarr paths")

        # Load zarr arrays and get volume shape
        arrays = [zarr.open(path, mode='r') for path in zarr_paths]
        self.shape = arrays[0].shape

        # Validate coordinates
        self.start = self._validate_start(start_coords)
        self.size = self._validate_size(start_coords, size_coords)

        # Load chunks and combine into RGB volume
        self.volume = self._load_rgb_volume(arrays, sigma)
        self.threshold = threshold

        # Initialize viewer and create visualization
        self.viewer = napari.Viewer(ndisplay=3)
        self._create_layers()
        self._add_controls()
        self._add_axes()

    def _validate_start(self, start):
        """Validate start coordinates are within volume bounds."""
        if len(start) != 3:
            raise ValueError(f"Expected (z,y,x) start coordinates, got {len(start)}")

        for i, (pos, size) in enumerate(zip(start, self.shape)):
            if pos < 0 or pos >= size:
                raise ValueError(f"Start position {pos} is outside volume bounds [0,{size}) in dimension {i}")

        return start

    def _validate_size(self, start, size):
        """Validate and adjust size to fit within volume bounds."""
        if len(size) != 3:
            raise ValueError(f"Expected (z,y,x) size coordinates, got {len(size)}")

        adjusted = []
        for i, (begin, length, total) in enumerate(zip(start, size, self.shape)):
            if length <= 0:
                raise ValueError(f"Size must be positive, got {length} in dimension {i}")
            adjusted.append(min(length, total - begin))

        return tuple(adjusted)

    def _load_rgb_volume(self, arrays, sigma):
        """
        Load and combine zarr arrays into a single RGB volume.

        Returns:
            numpy array of shape (D,H,W,3) with dtype uint8
        """
        channels = []
        for arr in arrays:
            # Extract chunk
            chunk = arr[
                    self.start[0]:self.start[0] + self.size[0],
                    self.start[1]:self.start[1] + self.size[1],
                    self.start[2]:self.start[2] + self.size[2]
                    ]

            channels.append(chunk)

        # Stack R,G,B channels
        rgb = np.stack(channels, axis=-1)  # Shape: (D,H,W,3)
        return rgb

    def _create_layers(self):
        """Create individual channel layers for RGB visualization."""
        colors = [('red', 'Red'), ('green', 'Green'), ('blue', 'Blue')]
        self.layers = []

        for i, (color, name) in enumerate(colors):
            layer = self.viewer.add_image(
                self.volume[..., i],  # Individual channel
                name=f"{name} Channel",
                blending='additive',
                rendering='attenuated_mip',
                colormap=color,
                visible=True,
                attenuation=0.01,
                contrast_limits=[self.threshold, 255],
                gamma=0.85
            )
            self.layers.append(layer)

    def _add_controls(self):
        """Add GUI controls for visualization parameters."""

        @magicgui(
            auto_call=True,
            red_threshold={"widget_type": "SpinBox",
                           "min": 0,
                           "max": 255,
                           "label": "Red Threshold",
                           "value": self.threshold},
            green_threshold={"widget_type": "SpinBox",
                             "min": 0,
                             "max": 255,
                             "label": "Green Threshold",
                             "value": self.threshold},
            blue_threshold={"widget_type": "SpinBox",
                            "min": 0,
                            "max": 255,
                            "label": "Blue Threshold",
                            "value": self.threshold},
            red_attenuation={"widget_type": "FloatSpinBox",
                             "min": -2,
                             "max": 2,
                             "label": "Red Attenuation",
                             "value": 0.1,
                             "step": 0.1},
            green_attenuation={"widget_type": "FloatSpinBox",
                               "min": -2,
                               "max": 2,
                               "label": "Green Attenuation",
                               "value": 0.1,
                               "step": 0.1},
            blue_attenuation={"widget_type": "FloatSpinBox",
                              "min": -2,
                              "max": 2,
                              "label": "Blue Attenuation",
                              "value": 0.1,
                              "step": 0.1},
            gamma={"widget_type": "FloatSpinBox",
                   "min": 0.1,
                   "max": 5.0,
                   "label": "Gamma",
                   "value": 0.85,
                   "step": 0.05}
        )
        def control_panel(
                red_threshold: int = self.threshold,
                green_threshold: int = self.threshold,
                blue_threshold: int = self.threshold,
                red_attenuation: float = 0.01,
                green_attenuation: float = 0.01,
                blue_attenuation: float = 0.01,
                gamma: float = 0.85
        ):
            # Update thresholds
            thresholds = [red_threshold, green_threshold, blue_threshold]
            attenuations = [red_attenuation, green_attenuation, blue_attenuation]

            for layer, threshold, attenuation in zip(self.layers, thresholds, attenuations):
                layer.contrast_limits = [threshold, 255]
                layer.attenuation = attenuation
                layer.gamma = gamma

        self.viewer.window.add_dock_widget(control_panel)

    def _add_axes(self):
        """Add axes and orientation indicators to the viewer."""
        # Add axes to viewer
        self.viewer.axes.visible = True
        self.viewer.axes.colored = True
        self.viewer.axes.labels = True
        self.viewer.axes.dashed = True

        # Add scale bar
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.unit = "px"

        # NOTE: In napari, dimensions are displayed as (0, 1, 2)
        # But they correspond to the array dimensions (z, y, x)
        # However, in the 3D view, these often get mapped as:
        # Dimension 0 (z in our array) → X axis in display
        # Dimension 1 (y in our array) → Y axis in display
        # Dimension 2 (x in our array) → Z axis in display

        # Create points at origin and along axes
        origin = np.array([0, 0, 0])
        axis_0_point = np.array([self.size[0] * 0.2, 0, 0])  # Dimension 0 (z in array)
        axis_1_point = np.array([0, self.size[1] * 0.2, 0])  # Dimension 1 (y in array)
        axis_2_point = np.array([0, 0, self.size[2] * 0.2])  # Dimension 2 (x in array)

        # Create vectors for axes
        axis_0 = np.stack([origin, axis_0_point])
        axis_1 = np.stack([origin, axis_1_point])
        axis_2 = np.stack([origin, axis_2_point])

        # Add vectors as separate shape layers
        self.viewer.add_shapes(
            axis_0,
            shape_type='line',
            edge_color='red',
            edge_width=5,
            name='Dimension 0 (+Z in array)'
        )

        self.viewer.add_shapes(
            axis_1,
            shape_type='line',
            edge_color='green',
            edge_width=5,
            name='Dimension 1 (+Y in array)'
        )

        self.viewer.add_shapes(
            axis_2,
            shape_type='line',
            edge_color='blue',
            edge_width=5,
            name='Dimension 2 (+X in array)'
        )

        # Add text labels
        labels = np.array([
            [self.size[0] * 0.25, 0, 0],  # Dimension 0
            [0, self.size[1] * 0.25, 0],  # Dimension 1
            [0, 0, self.size[2] * 0.25],  # Dimension 2
        ])

        texts = ['Dim 0 (Z)', 'Dim 1 (Y)', 'Dim 2 (X)']

        self.viewer.add_points(
            labels,
            name='Dimension Labels',
            text={
                'string': texts,
                'color': 'white',
                'size': 12
            },
            face_color='transparent',
            opacity=1.0,
            size=0
        )

        # Print dimensional information
        print("\nDimensional information:")
        print(f"Dimension 0 corresponds to Z in the array (shape: {self.volume.shape[0]})")
        print(f"Dimension 1 corresponds to Y in the array (shape: {self.volume.shape[1]})")
        print(f"Dimension 2 corresponds to X in the array (shape: {self.volume.shape[2]})")
        print("\nNOTE: In the napari 3D view, these dimensions may be displayed differently.")
        print("Use the dimension slider controls to confirm which slider affects which dimension.")

        # Print navigation help
        print("\nNavigation tips:")
        print("- Use mouse to rotate view (hold left button and drag)")
        print("- Use mouse wheel to zoom in/out")
        print("- Hold Shift + left mouse button to pan")
        print("- Press '3' to reset view")
        print("- Press '2' to toggle between 2D/3D view")

    def run(self):
        """Start the viewer."""
        # Print some guidance for the viewer
        print(f"Loaded volume with dimensions: {self.volume.shape}")
        print(f"Starting at coordinates: {self.start}")
        print(f"Size of chunk: {self.size}")
        napari.run()


if __name__ == "__main__":
    # Example usage
    paths = [
        "D:/53kev_1",  # Red channel
        "D:/70kev_1",  # Green channel
        "D:/88kev_1"  # Blue channel
    ]

    start = (3072-256, 0, 2048-256)  # Starting coordinates (z,y,x)
    size = (1024, 1024, 1024)  # Chunk size (z,y,x)
    threshold = 32  # Visualization threshold

    viewer = RGBVolumeViewer(paths, start, size, threshold)
    viewer.run()
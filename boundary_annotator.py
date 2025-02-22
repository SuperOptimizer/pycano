import sys
import numpy as np
import zarr
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QSlider,
                             QLabel, QSpinBox)
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, pyqtSignal
import json
from skimage import draw
from scipy.ndimage import binary_fill_holes
from pathlib import Path
import os


class ZarrCanvas(QWidget):
    """Canvas widget for displaying a zarr slice and capturing click events"""

    pointClicked = pyqtSignal(int, int)  # Signal to emit click coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.scale_factor = 1.0
        self.click_points = []  # List of (x, y) points that were clicked
        self.completed_outlines = {}  # Dict of z_index -> list of points
        self.current_z = 0
        self.marker_radius = 5
        self.setMinimumSize(400, 400)

    def set_image(self, data):
        """Set the current slice image data (2D numpy array)"""
        if data is None:
            self.image = None
            return

        # Convert to 8-bit for display if needed
        if data.dtype != np.uint8:
            # Normalize to 0-255 range
            min_val = data.min()
            max_val = data.max()
            if max_val > min_val:
                norm_data = ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                norm_data = np.zeros_like(data, dtype=np.uint8)
        else:
            norm_data = data

        # Create QImage from numpy array
        height, width = norm_data.shape
        bytes_per_line = width
        self.image = QImage(norm_data.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        self.update()

    def set_z_index(self, z):
        """Set the current z-index and update points display"""
        # Save current outline points if they exist
        if self.click_points and len(self.click_points) > 0:
            self.completed_outlines[self.current_z] = self.click_points.copy()

        # Update current z and load points for this frame
        self.current_z = z
        self.click_points = self.completed_outlines.get(z, []).copy()
        self.update()

    def add_point(self, x, y):
        """Add a clicked point to the current frame"""
        self.click_points.append((x, y))
        self.update()

    def clear_points(self):
        """Clear all points for the current frame"""
        self.click_points = []
        self.update()

    def save_current_outline(self):
        """Save the current outline points"""
        if len(self.click_points) >= 3:
            self.completed_outlines[self.current_z] = self.click_points.copy()
            return True
        return False

    def delete_current_outline(self):
        """Delete the outline for the current frame"""
        if self.current_z in self.completed_outlines:
            del self.completed_outlines[self.current_z]
            self.click_points = []
            self.update()
            return True
        return False

    def get_all_outlines(self):
        """Return all completed outlines"""
        return self.completed_outlines

    def mousePressEvent(self, event):
        """Handle mouse press events to capture clicks"""
        if self.image is None:
            return

        # Get click position and scale to image coordinates
        pos = event.pos()
        x = int(pos.x() / self.scale_factor)
        y = int(pos.y() / self.scale_factor)

        # Check if within image bounds
        if 0 <= x < self.image.width() and 0 <= y < self.image.height():
            self.add_point(x, y)
            self.pointClicked.emit(x, y)

    def paintEvent(self, event):
        """Draw the image and overlay points"""
        painter = QPainter(self)

        # Fill background
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        if self.image:
            # Calculate scaled dimensions
            scaled_width = int(self.image.width() * self.scale_factor)
            scaled_height = int(self.image.height() * self.scale_factor)

            # Draw the image
            pixmap = QPixmap.fromImage(self.image)
            scaled_pixmap = pixmap.scaled(scaled_width, scaled_height, Qt.KeepAspectRatio)
            painter.drawPixmap(0, 0, scaled_pixmap)

            # Draw points
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QBrush(QColor(255, 0, 0, 128)))

            for x, y in self.click_points:
                scaled_x = int(x * self.scale_factor)
                scaled_y = int(y * self.scale_factor)
                painter.drawEllipse(QPoint(scaled_x, scaled_y),
                                    self.marker_radius,
                                    self.marker_radius)

            # Draw lines connecting the points
            if len(self.click_points) > 1:
                painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
                for i in range(len(self.click_points) - 1):
                    x1, y1 = self.click_points[i]
                    x2, y2 = self.click_points[i + 1]
                    painter.drawLine(
                        int(x1 * self.scale_factor),
                        int(y1 * self.scale_factor),
                        int(x2 * self.scale_factor),
                        int(y2 * self.scale_factor)
                    )

                # Connect last point to first to close the polygon
                if len(self.click_points) >= 3:
                    x1, y1 = self.click_points[-1]
                    x2, y2 = self.click_points[0]
                    painter.drawLine(
                        int(x1 * self.scale_factor),
                        int(y1 * self.scale_factor),
                        int(x2 * self.scale_factor),
                        int(y2 * self.scale_factor)
                    )


class ZarrViewer(QMainWindow):
    """Main window for the Zarr viewer application"""

    def __init__(self):
        super().__init__()

        self.zarr_array = None
        self.mask = None

        self.init_ui()

        # Set focus policy to accept keyboard focus
        self.setFocusPolicy(Qt.StrongFocus)

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('Zarr Frame Outliner')
        self.resize(800, 600)

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Create canvas for displaying zarr data
        self.canvas = ZarrCanvas()
        main_layout.addWidget(self.canvas, 3)  # Canvas gets more space

        # Create control panel
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel, 1)  # Controls get less space

        # Add load button
        load_btn = QPushButton('Load Zarr')
        load_btn.clicked.connect(self.load_zarr)
        control_layout.addWidget(load_btn)

        # Add z-slice navigation
        z_nav_layout = QHBoxLayout()
        z_nav_layout.addWidget(QLabel('Z-Slice:'))

        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(0)
        self.z_slider.valueChanged.connect(self.change_z_slice)
        z_nav_layout.addWidget(self.z_slider)

        self.z_label = QLabel('0/0')
        z_nav_layout.addWidget(self.z_label)

        control_layout.addLayout(z_nav_layout)

        # Add frame jump controls
        jump_layout = QHBoxLayout()
        jump_layout.addWidget(QLabel('Jump:'))

        self.jump_amount = QSpinBox()
        self.jump_amount.setMinimum(1)
        self.jump_amount.setMaximum(100)
        self.jump_amount.setValue(5)
        jump_layout.addWidget(self.jump_amount)

        jump_prev = QPushButton('←')
        jump_prev.clicked.connect(self.jump_backward)
        jump_layout.addWidget(jump_prev)

        jump_next = QPushButton('→')
        jump_next.clicked.connect(self.jump_forward)
        jump_layout.addWidget(jump_next)

        control_layout.addLayout(jump_layout)

        # Add outline controls
        control_layout.addWidget(QLabel('Outline Controls:'))

        finish_btn = QPushButton('Finish Outline')
        finish_btn.clicked.connect(self.finish_outline)
        control_layout.addWidget(finish_btn)

        clear_btn = QPushButton('Clear Points')
        clear_btn.clicked.connect(self.clear_points)
        control_layout.addWidget(clear_btn)

        delete_btn = QPushButton('Delete Outline')
        delete_btn.clicked.connect(self.delete_outline)
        control_layout.addWidget(delete_btn)

        # Add processing controls
        control_layout.addWidget(QLabel('Processing:'))

        process_btn = QPushButton('Process Outlines')
        process_btn.clicked.connect(self.process_outlines)
        control_layout.addWidget(process_btn)

        save_btn = QPushButton('Save Processed Zarr')
        save_btn.clicked.connect(self.save_processed_zarr)
        control_layout.addWidget(save_btn)

        export_btn = QPushButton('Export Outlines')
        export_btn.clicked.connect(self.export_outlines)
        control_layout.addWidget(export_btn)

        # Add status label
        self.status_label = QLabel('Ready')
        control_layout.addWidget(self.status_label)

        # Add a spacer to push everything up
        control_layout.addStretch()

    def load_zarr(self):
        """Load a Zarr dataset"""
        zarr_path = QFileDialog.getExistingDirectory(
            self, 'Select Zarr Directory'
        )

        if not zarr_path:
            return

        try:
            # Load the Zarr array
            self.zarr_array = zarr.open(zarr_path, mode='r')

            # Update UI
            self.z_slider.setMaximum(self.zarr_array.shape[0] - 1)
            self.z_label.setText(f'0/{self.zarr_array.shape[0] - 1}')

            # Display the first slice
            self.change_z_slice(0)

            self.status_label.setText(f'Loaded: {zarr_path}')

        except Exception as e:
            self.status_label.setText(f'Error loading Zarr: {str(e)}')

    def change_z_slice(self, z):
        """Change the displayed z-slice"""
        if self.zarr_array is None:
            return

        try:
            # Get the slice data
            slice_data = self.zarr_array[z]

            # Update the canvas
            self.canvas.set_image(slice_data)
            self.canvas.set_z_index(z)

            # Update the label
            self.z_label.setText(f'{z}/{self.z_slider.maximum()}')

            self.status_label.setText(f'Showing z-slice: {z}')

        except Exception as e:
            self.status_label.setText(f'Error displaying slice: {str(e)}')

    def jump_forward(self):
        """Jump forward by the specified number of frames"""
        if self.zarr_array is None:
            return

        jump_size = self.jump_amount.value()
        current_z = self.z_slider.value()
        new_z = min(current_z + jump_size, self.z_slider.maximum())
        self.z_slider.setValue(new_z)

    def jump_backward(self):
        """Jump backward by the specified number of frames"""
        if self.zarr_array is None:
            return

        jump_size = self.jump_amount.value()
        current_z = self.z_slider.value()
        new_z = max(current_z - jump_size, 0)
        self.z_slider.setValue(new_z)

    def finish_outline(self):
        """Mark the current outline as complete"""
        if len(self.canvas.click_points) >= 3:
            current_z = self.z_slider.value()
            # Just confirm to the user it's ready
            self.status_label.setText(f'Outline completed for z={current_z}')
        else:
            self.status_label.setText('Need at least 3 points for an outline')

    def clear_points(self):
        """Clear the current points"""
        self.canvas.clear_points()
        self.status_label.setText('Points cleared')

    def delete_outline(self):
        """Delete the outline for the current frame"""
        if self.canvas.delete_current_outline():
            current_z = self.z_slider.value()
            self.status_label.setText(f'Outline deleted for z={current_z}')
        else:
            self.status_label.setText('No outline to delete')

    def process_outlines(self):
        """Process outlines to create a 3D mask with interpolation"""
        if self.zarr_array is None:
            self.status_label.setText('No Zarr data loaded')
            return

        outlines = self.canvas.get_all_outlines()
        if not outlines:
            self.status_label.setText('No outlines created yet')
            return

        try:
            # Create a mask for the entire volume
            self.mask = np.zeros(self.zarr_array.shape, dtype=bool)

            # Create masks for frames with defined outlines
            self.status_label.setText('Creating masks from outlines...')
            for z, points in outlines.items():
                if len(points) < 3:
                    continue

                # Create a binary mask for this slice
                slice_shape = self.zarr_array.shape[1:]  # YX dimensions
                mask_2d = np.zeros(slice_shape, dtype=bool)

                # Extract x, y coordinates
                polygon_y = [y for x, y in points]
                polygon_x = [x for x, y in points]

                # Fill the polygon
                rr, cc = draw.polygon(polygon_y, polygon_x, slice_shape)
                if len(rr) > 0 and len(cc) > 0:  # Ensure valid polygon
                    mask_2d[rr, cc] = True
                    mask_2d = binary_fill_holes(mask_2d)
                    self.mask[int(z)] = mask_2d

            # Interpolate masks between frames with outlines
            self.status_label.setText('Interpolating between outlined frames...')
            z_indices = sorted(list(outlines.keys()))

            if len(z_indices) > 1:
                for i in range(len(z_indices) - 1):
                    z_start = int(z_indices[i])
                    z_end = int(z_indices[i + 1])

                    if z_end - z_start > 1:  # If there are frames to interpolate
                        # Linear interpolation between masks
                        for z in range(z_start + 1, z_end):
                            weight_end = (z - z_start) / (z_end - z_start)
                            weight_start = 1 - weight_end

                            # Convert to float for interpolation
                            mask_start = self.mask[z_start].astype(float)
                            mask_end = self.mask[z_end].astype(float)

                            # Interpolate and threshold
                            interpolated = weight_start * mask_start + weight_end * mask_end
                            self.mask[z] = interpolated > 0.5

            self.status_label.setText('Processing complete! Mask created successfully.')

        except Exception as e:
            self.status_label.setText(f'Error processing outlines: {str(e)}')

    def save_processed_zarr(self):
        """Save the processed Zarr with the mask applied"""
        if self.zarr_array is None or self.mask is None:
            self.status_label.setText('Process outlines first')
            return

        # Get output directory
        output_path = QFileDialog.getExistingDirectory(
            self, 'Select Output Zarr Directory'
        )

        if not output_path:
            return

        try:
            self.status_label.setText('Saving masked Zarr...')

            # Create output zarr array
            out_array = zarr.open(
                output_path,
                mode='w',
                shape=self.zarr_array.shape,
                dtype=self.zarr_array.dtype,
                chunks=self.zarr_array.chunks if hasattr(self.zarr_array, 'chunks') else None
            )

            # Process each slice
            total_slices = self.zarr_array.shape[0]
            for z in range(total_slices):
                # Update status periodically
                if z % 10 == 0:
                    self.status_label.setText(f'Saving: {z}/{total_slices} slices')
                    QApplication.processEvents()  # Allow UI updates

                # Get the slice data
                slice_data = self.zarr_array[z]

                # Apply mask to this slice
                if z < self.mask.shape[0]:
                    masked_data = np.where(self.mask[z], slice_data, 0)
                else:
                    masked_data = slice_data

                # Write to output
                out_array[z] = masked_data

            self.status_label.setText(f'Masked Zarr saved to {output_path}')

        except Exception as e:
            self.status_label.setText(f'Error saving Zarr: {str(e)}')

    def keyPressEvent(self, event):
        """Handle keyboard events"""
        if event.key() == Qt.Key_Right:
            self.jump_forward()
        elif event.key() == Qt.Key_Left:
            self.jump_backward()
        else:
            super().keyPressEvent(event)

    def export_outlines(self):
        """Export outlines to a JSON file"""
        outlines = self.canvas.get_all_outlines()
        if not outlines:
            self.status_label.setText('No outlines to export')
            return

        # Get output file
        output_file, _ = QFileDialog.getSaveFileName(
            self, 'Save Outlines JSON', '', 'JSON Files (*.json)'
        )

        if not output_file:
            return

        try:
            # Convert outlines to a serializable format
            serialized_outlines = {}
            for z, points in outlines.items():
                # Convert to string keys for JSON
                z_key = str(int(z))

                # Convert points to x,y dictionary format
                serialized_outlines[z_key] = [
                    {"x": float(x), "y": float(y)} for x, y in points
                ]

            # Save to file
            with open(output_file, 'w') as f:
                json.dump(serialized_outlines, f, indent=2)

            self.status_label.setText(f'Outlines exported to {output_file}')

        except Exception as e:
            self.status_label.setText(f'Error exporting outlines: {str(e)}')


def main():
    app = QApplication(sys.argv)
    viewer = ZarrViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
import numpy as np
from skimage import exposure, filters, restoration, transform
from scipy import ndimage
import zarr
import requests
import blosc2
from typing import Tuple, Optional
import napari
import skimage
import glcae

RESOLUTION_SPECS = {
    4: {'dtype': np.uint8, 'max_val': 15},
    8: {'dtype': np.uint8, 'max_val': 127},
    16: {'dtype': np.uint16, 'max_val': 1024},
    32: {'dtype': np.uint16, 'max_val': 8192},
    64: {'dtype': np.uint16, 'max_val': 65535},
    128: {'dtype': np.uint32, 'max_val': 524288}
}


def process_volume(volume, input_resolution, target_resolution=None, denoise_strength=0.1, enhance_contrast=True):
    vol = volume.astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min())

    vol = restoration.denoise_tv_chambolle(vol, weight=denoise_strength)

    if enhance_contrast:
        p2, p98 = np.percentile(vol, (1, 99))
        vol = np.clip((vol - p2) / (p98 - p2), 0, 1)

        uint8_vol = (vol * 255).astype(np.uint8)
        vol = glcae.global_and_local_contrast_enhancement_3d(uint8_vol) / 255

        gx = ndimage.gaussian_filter1d(vol, 1.0, axis=2)
        gy = ndimage.gaussian_filter1d(vol, 1.0, axis=1)
        gz = ndimage.gaussian_filter1d(vol, 1.0, axis=0)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min())

        vol = np.clip(vol + 0.5 * grad_mag, 0, 1)
        vol = exposure.adjust_sigmoid(vol, cutoff=0.5, gain=10)

    if target_resolution is not None:
        scale = input_resolution / target_resolution
        if scale < 1.0:
            sigma = 0.5 * (1.0 / scale - 1.0)
            if sigma > 0:
                vol = ndimage.gaussian_filter(vol, sigma)
            vol = transform.resize(
                vol,
                tuple(int(s * scale) for s in vol.shape),
                order=3,
                mode='reflect'
            )

    return np.clip(vol, 0, 1)


class XrayDataset:
    def __init__(self, store_path):
        self.store = zarr.open(store_path, mode='w')

    def save_volume(self, volume, input_resolution, resolution, chunk_size=(128, 128, 128)):
        spec = RESOLUTION_SPECS[resolution]
        processed = process_volume(volume, input_resolution=input_resolution, target_resolution=resolution)
        normalized = (processed * spec['max_val']).astype(spec['dtype'])
        group = self.store.create_group(f'resolution_{resolution}um')
        group.create_dataset('data', data=normalized, chunks=chunk_size, dtype=spec['dtype'])

    def close(self):
        self.store.close()


def get_chunk_url(base_url: str, chunk_coords: Tuple[int, int, int]) -> str:
    return f"{base_url}/{chunk_coords[0]}/{chunk_coords[1]}/{chunk_coords[2]}"


def download_chunk(url: str) -> Optional[bytes]:
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error downloading chunk: {e}")
        return None


def decompress_chunk(compressed_data: bytes, dtype: str = '|u1') -> np.ndarray:
    decompressed = blosc2.decompress(compressed_data)
    array = np.frombuffer(decompressed, dtype=dtype)
    return array.reshape((128, 128, 128))


def fetch_volume_chunk(base_url: str, z: int, y: int, x: int, dtype: str = '|u1') -> Optional[np.ndarray]:
    chunk_url = get_chunk_url(base_url, (z, y, x))
    compressed = download_chunk(chunk_url)
    if compressed is None: return None
    try:
        return decompress_chunk(compressed, dtype)
    except Exception as e:
        print(f"Error decompressing chunk: {e}")
        return None


def get_chunk_indices(pos: Tuple[int, int, int], chunk_size: Tuple[int, int, int] = (128, 128, 128)) -> Tuple[
    int, int, int]:
    return tuple(p // c for p, c in zip(pos, chunk_size))


def show_volume(volume):
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(
        volume,
        colormap='gray',
        rendering='iso',
        iso_threshold=0.35
    )

    verts, faces, normals, values = skimage.measure.marching_cubes(volume, level=0.35)
    viewer.add_surface((verts, faces), opacity=0.9)
    napari.run()


if __name__ == "__main__":
    base_url = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0/"
    chunk_idx = (50, 40, 40)
    chunk = fetch_volume_chunk(base_url, *chunk_idx)
    input_resolution = 3.24

    for target_resolution in [4, 8, 16, 32, 64, 128]:
        if input_resolution > target_resolution:
            continue
        processed = process_volume(chunk, input_resolution=input_resolution, target_resolution=target_resolution)
        show_volume(processed)
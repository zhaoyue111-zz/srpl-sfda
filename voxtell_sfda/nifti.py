from __future__ import annotations

from pathlib import Path

import nibabel
import numpy as np
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform


def write_reoriented_nifti(array: np.ndarray, output_path: str | Path, properties: dict, dtype=np.float32) -> None:
    """Write a 3D array with the orientation metadata returned by nnU-Net's reader."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = array.astype(dtype, copy=False).transpose((2, 1, 0))
    image = nibabel.Nifti1Image(data, affine=properties["nibabel_stuff"]["reoriented_affine"])

    image_orientation = io_orientation(properties["nibabel_stuff"]["original_affine"])
    ras_orientation = axcodes2ornt("RAS")
    from_canonical = ornt_transform(ras_orientation, image_orientation)
    image_reoriented = image.as_reoriented(from_canonical)
    if not np.allclose(properties["nibabel_stuff"]["original_affine"], image_reoriented.affine):
        print(f"WARNING: Restored affine does not match original affine. File: {output_path}")
        print("Original affine\n", properties["nibabel_stuff"]["original_affine"])
        print("Restored affine\n", image_reoriented.affine)
    nibabel.save(image_reoriented, str(output_path))

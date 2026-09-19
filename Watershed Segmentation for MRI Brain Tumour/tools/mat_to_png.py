"""
Convert the Figshare brain-tumour dataset (.mat files) into PNG image/mask pairs.

Dataset: Jun Cheng, "brain tumor dataset", figshare, 2017.
https://doi.org/10.6084/m9.figshare.1512427
3064 T1-weighted contrast-enhanced slices from 233 patients with meningioma,
glioma or pituitary tumour.

Every <id>.mat contains a struct `cjdata` with the fields
    image        512 x 512 MRI slice
    tumorMask    binary tumour mask
    label        1 = meningioma, 2 = glioma, 3 = pituitary
    PID          patient ID
    tumorBorder  tumour outline coordinates (not used)

For each file the script writes <id>_image.png and <id>_mask.png into one
folder and lists all slices in a metadata CSV.

Usage:
    python tools/mat_to_png.py                  # data/mat -> data/png
    python tools/mat_to_png.py --mat-dir D:/brain_tumor/mat --out-dir data/png
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent.parent
LABELS = {1.0: "meningioma", 2.0: "glioma", 3.0: "pituitary"}


def read_mat(mat_path):
    """
    Read one dataset file.

    The files are MATLAB v7.3 (HDF5) and are read with h5py; older MATLAB
    formats fall back to scipy.io.loadmat.

    Returns:
        (image, tumor_mask, label_id, patient_id)
    """
    try:
        with h5py.File(mat_path, 'r') as f:
            cjdata = f['cjdata']
            # HDF5 stores MATLAB arrays column-major, hence the transpose.
            image = np.array(cjdata['image']).T
            tumor_mask = np.array(cjdata['tumorMask']).T
            label_id = float(np.array(cjdata['label']).flat[0])
            # PID is stored either as uint16 character codes or as a string.
            pid_arr = np.array(cjdata['PID'])
            if pid_arr.dtype.kind in ('u', 'i'):
                patient_id = "".join(chr(int(c)) for c in pid_arr.flat).strip()
            else:
                patient_id = str(pid_arr.flat[0]).strip()
    except Exception:
        cjdata = loadmat(mat_path)['cjdata'][0, 0]
        image = cjdata['image']
        tumor_mask = cjdata['tumorMask']
        label_id = float(np.array(cjdata['label']).flat[0])
        # PID arrives as an object array of strings / char arrays.
        patient_id = str(np.array(cjdata['PID']).flat[0]).strip("[]' ")

    return image, tumor_mask, label_id, patient_id


def main():
    parser = argparse.ArgumentParser(
        description="Convert the Figshare brain-tumour .mat files to PNG.")
    parser.add_argument("--mat-dir", type=Path, default=ROOT / "data" / "mat",
                        help="folder with the <id>.mat files")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "png",
                        help="output folder for <id>_image.png / <id>_mask.png")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "dataset_metadata.csv",
                        help="output metadata CSV")
    args = parser.parse_args()

    mat_files = sorted(args.mat_dir.glob("*.mat"))
    if not mat_files:
        raise SystemExit(f"No .mat files found in {args.mat_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(mat_files)} files into {args.out_dir} ...")
    records = []
    for i, mat_path in enumerate(mat_files, 1):
        file_id = mat_path.stem
        try:
            image, tumor_mask, label_id, patient_id = read_mat(mat_path)
        except Exception as e:
            print(f"Skipping {mat_path.name}: could not parse ({e})")
            continue

        image_name = f"{file_id}_image.png"
        mask_name = f"{file_id}_mask.png"
        try:
            # imsave min-max scales each slice to the full grey range (RGBA PNG).
            plt.imsave(args.out_dir / image_name, image, cmap='gray')
            plt.imsave(args.out_dir / mask_name, tumor_mask, cmap='gray')
        except Exception as e:
            print(f"Error saving PNGs for {mat_path.name}: {e}")
            continue

        records.append({
            "file_id": file_id,
            "patient_id": patient_id,
            "label_id": int(label_id),
            "tumor_type": LABELS.get(label_id, "unknown"),
            "image_filename": image_name,
            "mask_filename": mask_name,
        })
        if i % 250 == 0:
            print(f"  {i}/{len(mat_files)}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.csv, index=False)
    print(f"Done: {len(records)} slices converted, metadata written to {args.csv}")


if __name__ == "__main__":
    main()

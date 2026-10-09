"""
Evaluate a trained model on PancreasCT with sliding-window inference and report
Dice, HD95 and ASD before and after Largest-Connected-Component (LCC) post-processing.

The default list is test.list (PANCREAS_0063-0082), scans never used in training.
The thesis numbers were computed on test1.list (PANCREAS_0001-0020), which
overlaps train.list; pass --test_list test1.list to reproduce them.

Run from the code/ directory:
    python evaluate_pancreas_lcc.py --model_type unet_3D --model_path ../models/<exp>/<run>/unet_3D_best_model.pth
"""
import os
import csv
import argparse

import numpy as np

# medpy 0.4 still references np.bool and np.float_, which recent NumPy versions removed.
if not hasattr(np, 'bool'): np.bool = np.bool_
if not hasattr(np, 'float_'): np.float_ = np.float64

import h5py
import nibabel as nib
import torch
from medpy import metric
from skimage.measure import label
from tqdm import tqdm

from networks.net_factory_3d import net_factory_3d
from utils.test_3d_patch import test_single_case


def get_lcc(prediction):
    """Keep only the largest 3D connected component of a binary mask.

    skimage.measure.label uses full connectivity by default, i.e. 26-connectivity in 3D.
    """
    labeled_array, num_features = label(prediction, return_num=True)
    if num_features == 0:
        return prediction
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0  # ignore the background
    lcc_label = sizes.argmax()
    return (labeled_array == lcc_label).astype(prediction.dtype)


def calculate_metrics(pred, gt):
    """Dice (%), HD95 and ASD (in voxels) for one case.

    Empty masks get fixed scores instead of an error: 100 / 0 / 0 when both
    masks are empty, 0 / 500 / 500 when only the prediction is empty or medpy
    cannot compute a surface distance.
    """
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    if np.sum(pred) == 0:
        if np.sum(gt) == 0:
            return 100.0, 0.0, 0.0
        return 0.0, 500.0, 500.0

    try:
        dice = metric.binary.dc(pred, gt) * 100.0
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
    except RuntimeError:
        # Raised when the ground truth is empty but the prediction is not.
        return 0.0, 500.0, 500.0

    return dice, hd95, asd


def print_summary(title, dices, hd95s, asds):
    print(title)
    print(f"   Dice (%) : {np.mean(dices):.2f} ± {np.std(dices):.2f}")
    print(f"   HD95     : {np.mean(hd95s):.2f} ± {np.std(hd95s):.2f}")
    print(f"   ASD      : {np.mean(asds):.2f} ± {np.std(asds):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a 3D segmentation model with LCC post-processing.")
    parser.add_argument('--model_path', type=str, required=True, help='Checkpoint (.pth) to evaluate')
    parser.add_argument('--model_type', type=str, default='swin_umamba', choices=['swin_umamba', 'unet_3D'], help='Backbone')
    parser.add_argument('--data_root', type=str, default='../data/Pancreas', help='Folder with the .list files and Pancreas_data/')
    parser.add_argument('--test_list', type=str, default='test.list',
                        help='Scans to evaluate: test.list is held out; test1.list reproduces the thesis protocol')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64], help='Sliding-window size')
    parser.add_argument('--stride_xy', type=int, default=32, help='Sliding-window stride in the first two axes')
    parser.add_argument('--stride_z', type=int, default=32, help='Sliding-window stride in the third axis')
    parser.add_argument('--scaler', type=int, default=2, help='Feature scaler the model was trained with')
    parser.add_argument('--chunk_size', type=int, default=8, help='Swin-UMamba: 2D slices per forward chunk')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='If set, save image, ground truth, RAW / LCC predictions and the probability map as NIfTI')
    parser.add_argument('--csv', type=str, default=None, help='If set, write per-case metrics to this CSV file')
    args = parser.parse_args()

    print(f"Loading weights from: {args.model_path}")
    print(f"Evaluation params: patch_size={args.patch_size}, stride_xy={args.stride_xy}, stride_z={args.stride_z}")

    net = net_factory_3d(
        net_type=args.model_type,
        in_chns=1,
        class_num=2,
        scaler=args.scaler,
        chunk_size=args.chunk_size
    ).cuda()

    checkpoint = torch.load(args.model_path, map_location='cuda:0', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    net.eval()

    with open(os.path.join(args.data_root, args.test_list), 'r') as f:
        test_list = [line.strip() for line in f if line.strip()]

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    rows = []
    print(f"\nEvaluating {len(test_list)} scans from {args.test_list} (Dice, HD95, ASD)\n")

    for case in tqdm(test_list, ncols=80):
        with h5py.File(os.path.join(args.data_root, "Pancreas_data", case), 'r') as h5f:
            image = h5f['image'][:]
            gt = h5f['label'][:]

        prediction, score_map = test_single_case(net,
                                                 image,
                                                 stride_xy=args.stride_xy,
                                                 stride_z=args.stride_z,
                                                 patch_size=tuple(args.patch_size),
                                                 num_classes=2)

        raw = calculate_metrics(prediction, gt)
        prediction_lcc = get_lcc(prediction)
        lcc = calculate_metrics(prediction_lcc, gt)
        rows.append((case, *raw, *lcc))

        if args.save_dir:
            case_name = case.replace('.h5', '')
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(image.astype(np.float32), affine), os.path.join(args.save_dir, f"{case_name}_IMG.nii.gz"))
            nib.save(nib.Nifti1Image(gt.astype(np.uint8), affine), os.path.join(args.save_dir, f"{case_name}_GT.nii.gz"))
            nib.save(nib.Nifti1Image(prediction.astype(np.uint8), affine), os.path.join(args.save_dir, f"{case_name}_RAW.nii.gz"))
            nib.save(nib.Nifti1Image(prediction_lcc.astype(np.uint8), affine), os.path.join(args.save_dir, f"{case_name}_LCC.nii.gz"))
            nib.save(nib.Nifti1Image(score_map[1].astype(np.float32), affine), os.path.join(args.save_dir, f"{case_name}_PROB.nii.gz"))

    columns = list(zip(*rows))
    print("\n" + "=" * 50)
    print_summary("RAW metrics (before LCC):", columns[1], columns[2], columns[3])
    print("-" * 50)
    print_summary("LCC metrics (largest connected component):", columns[4], columns[5], columns[6])
    print("=" * 50 + "\n")

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['case', 'raw_dice', 'raw_hd95', 'raw_asd', 'lcc_dice', 'lcc_hd95', 'lcc_asd'])
            for case, *values in rows:
                writer.writerow([case] + [f"{v:.4f}" for v in values])
        print(f"Per-case metrics written to {args.csv}")


if __name__ == '__main__':
    main()

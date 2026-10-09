import os
import torch
import numpy as np

# Fix for Numpy 2.0 compatibility with Medpy
if not hasattr(np, 'bool'): np.bool = np.bool_
if not hasattr(np, 'float_'): np.float_ = np.float64

import argparse
import h5py
import nibabel as nib
from tqdm import tqdm
from skimage.measure import label
from medpy import metric
from networks.net_factory_3d import net_factory_3d
from utils.test_3d_patch import test_single_case

def get_lcc(prediction):
    labeled_array, num_features = label(prediction, return_num=True)
    if num_features == 0:
        return prediction
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0 
    lcc_label = sizes.argmax() 
    return (labeled_array == lcc_label).astype(prediction.dtype)

def calculate_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    
    # Διαχείριση οριακών περιπτώσεων με κενές μάσκες για αποφυγή σφαλμάτων.
    if np.sum(pred) == 0:
        if np.sum(gt) == 0:
            # Η πρόβλεψη και η αλήθεια είναι και οι δύο κενές -> τέλειο σκορ.
            return 100.0, 0.0, 0.0
        else:
            # Η πρόβλεψη είναι κενή αλλά η αλήθεια όχι -> το χειρότερο σκορ.
            # Χρησιμοποιούμε μια μεγάλη τιμή απόστασης για μεγάλη ποινή.
            return 0.0, 500.0, 500.0
    
    try:
        dice = metric.binary.dc(pred, gt) * 100.0
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
    except RuntimeError:
        # Αυτό συμβαίνει αν το ground truth είναι κενό αλλά η πρόβλεψη όχι.
        return 0.0, 500.0, 500.0

    return dice, hd95, asd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a 3D segmentation model with LCC post-processing.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file.')
    parser.add_argument('--model_type', type=str, default='swin_umamba', help='The architecture type (e.g., swin_umamba).')
    parser.add_argument('--data_root', type=str, default='../data/Pancreas', help='Root directory of the Pancreas dataset.')
    parser.add_argument('--test_list', type=str, default='test1.list', help='Name of the test list file.')
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64], help='Patch size for sliding window inference (H W D).')
    parser.add_argument('--stride_xy', type=int, default=32, help='Stride for sliding window inference in XY plane.')
    parser.add_argument('--stride_z', type=int, default=32, help='Stride for sliding window inference in Z axis.')
    # Model-specific args for SwinUMamba
    parser.add_argument('--scaler', type=int, default=2, help='Feature scaler for the model.')
    parser.add_argument('--chunk_size', type=int, default=8, help='Chunk size for SwinUMamba.')
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

    checkpoint = torch.load(args.model_path, map_location='cuda:0')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        net.load_state_dict(checkpoint)
    net.eval()

    with open(os.path.join(args.data_root, args.test_list), 'r') as f:
        test_list = [line.strip() for line in f if line.strip()]

    raw_dices, raw_hd95s, raw_asds = [], [], []
    lcc_dices, lcc_hd95s, lcc_asds = [], [], []

    save_dir = "./3D_Predictions_SwinUMamba"
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🚀 Starting Evaluation (Dice, HD95, ASD)...\n")
    
    for case in tqdm(test_list, ncols=80):
        data_path = os.path.join(args.data_root, "Pancreas_data", case)
        
        h5f = h5py.File(data_path, 'r')
        image = h5f['image'][:]
        gt = h5f['label'][:]
        
        prediction, score_map = test_single_case(net, 
                                         image, 
                                         stride_xy=args.stride_xy, 
                                         stride_z=args.stride_z, 
                                         patch_size=tuple(args.patch_size), 
                                         num_classes=2)
        
        r_dice, r_hd95, r_asd = calculate_metrics(prediction, gt)
        raw_dices.append(r_dice); raw_hd95s.append(r_hd95); raw_asds.append(r_asd)
        
        prediction_lcc = get_lcc(prediction)
        l_dice, l_hd95, l_asd = calculate_metrics(prediction_lcc, gt)
        lcc_dices.append(l_dice); lcc_hd95s.append(l_hd95); lcc_asds.append(l_asd)

        case_name = case.replace('.h5', '')
        nib.save(nib.Nifti1Image(prediction.astype(np.uint8), np.eye(4)), os.path.join(save_dir, f"{case_name}_RAW.nii.gz"))
        nib.save(nib.Nifti1Image(prediction_lcc.astype(np.uint8), np.eye(4)), os.path.join(save_dir, f"{case_name}_LCC.nii.gz"))
        nib.save(nib.Nifti1Image(gt.astype(np.uint8), np.eye(4)), os.path.join(save_dir, f"{case_name}_GT.nii.gz"))
        nib.save(nib.Nifti1Image(image.astype(np.float32), np.eye(4)), os.path.join(save_dir, f"{case_name}_IMG.nii.gz"))
        # Αποθήκευση του Softmax Probability Map (Heatmap ενεργοποίησης του παγκρέατος)
        nib.save(nib.Nifti1Image(score_map[1].astype(np.float32), np.eye(4)), os.path.join(save_dir, f"{case_name}_HEATMAP.nii.gz"))

    print("\n" + "="*50)
    print("📊 RAW Metrics (Πριν το LCC):")
    print(f"   Dice (%) : {np.mean(raw_dices):.2f} ± {np.std(raw_dices):.2f}")
    print(f"   HD95     : {np.mean(raw_hd95s):.2f} ± {np.std(raw_hd95s):.2f}")
    print(f"   ASD      : {np.mean(raw_asds):.2f} ± {np.std(raw_asds):.2f}")
    print("-" * 50)
    print("🏆 LCC Metrics (ΤΕΛΙΚΟ ΣΚΟΡ):")
    print(f"   Dice (%) : {np.mean(lcc_dices):.2f} ± {np.std(lcc_dices):.2f}")
    print(f"   HD95     : {np.mean(lcc_hd95s):.2f} ± {np.std(lcc_hd95s):.2f}")
    print(f"   ASD      : {np.mean(lcc_asds):.2f} ± {np.std(lcc_asds):.2f}")
    print("="*50 + "\n")
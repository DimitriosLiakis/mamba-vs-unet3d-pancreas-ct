"""Sliding-window inference and the validation loop used during training (adapted from the DyCON codebase)."""
import os
import math

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm


def var_all_case_Pancreas(model, root_path, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4,
                          list_name="test1.list"):
    """Mean Dice over the scans in `list_name`; used to pick the best checkpoint during training.

    The thesis runs used test1.list (PANCREAS_0001-0020), which overlaps train.list.
    """
    with open(os.path.join(root_path, list_name), "r") as f:
        case_ids = [line.strip() for line in f if line.strip()]
    image_list = [os.path.join(root_path, f"Pancreas_data/{case_id}") for case_id in case_ids]

    total_dice = 0.0
    for image_path in tqdm(image_list):
        with h5py.File(image_path, 'r') as h5f:
            image = h5f['image'][:]
            label = h5f['label'][:].astype(np.uint8)

        prediction, _ = test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction) == 0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction, label)
        total_dice += dice
    avg_dice = total_dice / len(image_list)
    print('average metric is {}'.format(avg_dice))
    return avg_dice


def test_single_case(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    """Sliding-window inference over one volume.

    The volume is padded if it is smaller than a patch, patches are taken with
    the given strides, and overlapping foreground probabilities are averaged.
    As in the original DyCON code, every channel of `score_map` accumulates the
    foreground probability and the label map thresholds it at 0.5, which for
    two classes is the same as an argmax.

    Returns:
        label_map: (W, H, D) binary prediction
        score_map: (num_classes, W, H, D) averaged foreground probability
    """
    w, h, d = image.shape

    # Pad the volume if it is smaller than the patch.
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    _, y, _ = model(test_patch)
                    y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, 1, :, :, :]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(int)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad+w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map, score_map

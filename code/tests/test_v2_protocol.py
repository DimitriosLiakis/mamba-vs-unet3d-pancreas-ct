"""
Checks of the v2 data protocol that need no GPU. Run from the code/ directory:
    python tests/test_v2_protocol.py

1. train_v2.list, val.list and test.list do not overlap, and train_v2.list + val.list is train.list.
2. With hide_unlabeled_labels the data loader returns all-zero labels for the unlabeled scans
   (needs the .h5 volumes in ../data/Pancreas/Pancreas_data and is skipped without them).
3. With source='teacher' the FeCL mask of the unlabeled scans does not depend on their
   ground-truth masks. With source='gt' it does.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloaders.pancreas import Pancreas
from utils.dycon_losses import build_fecl_mask

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'Pancreas')


def read_list(name):
    with open(os.path.join(DATA_ROOT, name)) as f:
        return [line.strip() for line in f if line.strip()]


def test_splits():
    train, val, test = read_list('train_v2.list'), read_list('val.list'), read_list('test.list')
    assert len(train) == 54 and len(val) == 8 and len(test) == 20
    assert not set(train) & set(val) and not set(train) & set(test) and not set(val) & set(test)
    assert train + val == read_list('train.list')
    print("splits: OK")


def test_hidden_labels(labelnum=12):
    if not os.path.isdir(os.path.join(DATA_ROOT, 'Pancreas_data')):
        print("hidden labels: skipped (no data)")
        return
    hidden = Pancreas(base_dir=DATA_ROOT, split='train', list_name='train_v2.list',
                      labeled_num=labelnum, hide_unlabeled_labels=True)
    shown = Pancreas(base_dir=DATA_ROOT, split='train', list_name='train_v2.list')
    assert hidden[0]['label'].sum() > 0 and hidden[labelnum - 1]['label'].sum() > 0
    for idx in (labelnum, labelnum + 1, len(hidden) - 1):
        assert hidden[idx]['label'].sum() == 0, f"unlabeled scan {idx} still has a mask"
        assert shown[idx]['label'].sum() > 0
        assert np.array_equal(hidden[idx]['image'], shown[idx]['image'])
    print("hidden labels: OK")


def test_fecl_mask(labeled_bs=1):
    torch.manual_seed(0)
    labels = torch.randint(0, 2, (2, 64, 64, 64))
    probs = torch.softmax(torch.randn(2, 2, 64, 64, 64), dim=1)
    other = labels.clone()
    other[labeled_bs:] = 1 - other[labeled_bs:]   # different ground truth for the unlabeled scan

    teacher = build_fecl_mask(labels, probs, labeled_bs, scale=1 / 8, source='teacher')
    assert teacher.shape == (2, 1, 8 * 8 * 8)
    assert torch.equal(teacher[labeled_bs:], build_fecl_mask(other, probs, labeled_bs, 1 / 8, 'teacher')[labeled_bs:])

    gt = build_fecl_mask(labels, probs, labeled_bs, scale=1 / 8, source='gt')
    assert torch.equal(teacher[:labeled_bs], gt[:labeled_bs])
    assert not torch.equal(gt[labeled_bs:], build_fecl_mask(other, probs, labeled_bs, 1 / 8, 'gt')[labeled_bs:])
    print("FeCL mask: OK")


if __name__ == '__main__':
    test_splits()
    test_hidden_labels()
    test_fecl_mask()

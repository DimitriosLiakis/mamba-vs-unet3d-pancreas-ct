"""Training-time Dice on binarized predictions (adapted from the DyCON codebase)."""


def compute_dice(output, label):
    """Per-sample Dice for binary tensors of shape (B, W, H, D)."""
    intersection = (output * label).sum(dim=(1, 2, 3))
    dice_score = (2. * intersection) / (output.sum(dim=(1, 2, 3)) + label.sum(dim=(1, 2, 3)))
    return dice_score

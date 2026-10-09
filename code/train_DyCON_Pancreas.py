"""
Train a segmentation network on PancreasCT with the DyCON semi-supervised
framework (mean teacher + UnCL + FeCL), or a fully supervised baseline.

Backbones: a 3D U-Net and Swin-UMamba (2D Mamba blocks applied slice by slice).
Setting --labeled_bs equal to --batch_size trains on the labeled scans only.

The defaults are the thesis configuration for a single 16 GB GPU: 64^3 patches,
1 labeled + 1 unlabeled scan per step, gradient accumulation over 4 steps,
AdamW at 1e-4 and 20k iterations. scripts/train_pancreas.sh also turns on the
VMamba encoder initialization for Swin-UMamba.

Run from the code/ directory:
    python train_DyCON_Pancreas.py --model unet_3D
"""
import os
import sys
import shutil
import random
import logging
import argparse
import numpy as np

# medpy 0.4 (used by the validation loop) still references np.bool and
# np.float_, which recent NumPy versions removed.
if not hasattr(np, 'bool'): np.bool = np.bool_
if not hasattr(np, 'float_'): np.float_ = np.float64

from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T

from networks.net_factory_3d import net_factory_3d
from utils import ramps, metrics, losses, dycon_losses, test_3d_patch
from dataloaders.pancreas import Pancreas, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

parser = argparse.ArgumentParser(description="Train DyCON (semi-supervised) or a fully supervised baseline on PancreasCT")

# Data and experiment
parser.add_argument('--root_dir', type=str, default="../data/Pancreas", help='Folder with train.list and Pancreas_data/*.h5')
parser.add_argument('--val_list', type=str, default='test1.list',
                    help='Scans used to select the best checkpoint. The thesis runs used test1.list, which overlaps train.list (see README)')
parser.add_argument('--exp', type=str, default='PancreasCT', help='Experiment name (output folder under ../models)')
parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64], help='Training patch size')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU to use')
parser.add_argument('--seed', type=int, default=1337, help='Random seed')
parser.add_argument('--deterministic', type=int, default=1, help='Deterministic cuDNN (0 or 1)')

# Model
parser.add_argument('--model', type=str, choices=['unet_3D', 'swin_umamba'], default='swin_umamba', help='Backbone')
parser.add_argument('--in_ch', type=int, default=1, help='Number of input channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes')
parser.add_argument('--feature_scaler', type=int, default=2,
                    help='Contrastive features are computed at 1/(4*feature_scaler) of the patch resolution')
parser.add_argument('--chunk_size', type=int, default=8, help='Swin-UMamba: 2D slices per forward chunk (lower uses less memory)')
parser.add_argument('--use_pretrain', type=int, default=0, help='Swin-UMamba: initialize the encoder from VMamba-tiny ImageNet weights')
parser.add_argument('--pretrained_ckpt', type=str, default='./data/pretrained/vmamba/vmamba_tiny_e292.pth', help='Path to the VMamba weights')

# Optimization
parser.add_argument('--max_iterations', type=int, default=20000, help='Number of training iterations')
parser.add_argument('--batch_size', type=int, default=2, help='Scans per step (labeled + unlabeled)')
parser.add_argument('--labeled_bs', type=int, default=1, help='Labeled scans per step; equal to --batch_size gives fully supervised training')
parser.add_argument('--labelnum', type=int, default=12, help='Number of labeled scans (the first N entries of train.list)')
parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'])
parser.add_argument('--base_lr', type=float, default=1e-4, help='Base learning rate')
parser.add_argument('--warmup_epochs', type=int, default=100, help='Linear warm-up epochs before cosine decay')
parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps (effective batch = batch_size * steps)')
parser.add_argument('--use_amp', type=int, default=0, help='Mixed precision (off for the thesis runs)')
parser.add_argument('--load_path', type=str, default=None,
                    help='Checkpoint to resume from (restores model, teacher, optimizer, scheduler and iteration)')
parser.add_argument('--keep_top_k', type=int, default=5, help='Number of best iter_*.pth checkpoints to keep on disk')

# DyCON
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay of the teacher')
parser.add_argument('--consistency', type=float, default=0.1, help='Weight of the consistency loss')
parser.add_argument('--consistency_type', type=str, default="mse", choices=['mse', 'kl'], help='Consistency loss type')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='Ramp-up length of the consistency weight')
parser.add_argument('--gamma', type=float, default=2.0, help='FeCL focal exponent')
parser.add_argument('--beta_min', type=float, default=0.5, help='UnCL beta schedule: minimum')
parser.add_argument('--beta_max', type=float, default=5.0, help='UnCL beta schedule: maximum')
parser.add_argument('--s_beta', type=float, default=None, help='Fixed UnCL beta (overrides the schedule)')
parser.add_argument('--temp', type=float, default=0.6, help='FeCL temperature')
parser.add_argument('--l_weight', type=float, default=1.0, help='Weight of the supervised loss')
parser.add_argument('--u_weight', type=float, default=0.5, help='Weight of UnCL + FeCL')
parser.add_argument('--use_focal', type=int, default=1, help='FeCL focal weighting')
parser.add_argument('--use_teacher_loss', type=int, default=1, help='FeCL: also contrast against teacher features')

args = parser.parse_args()

if args.s_beta is not None:
    beta_str = f"_beta{args.s_beta}"
else:
    beta_str = f"_beta{args.beta_min}-{args.beta_max}"

focal_str = "Focal" if bool(args.use_focal) else "NoFocal"
gamma_str = f"_gamma{args.gamma}" if bool(args.use_focal) else ""
teacher_str = "Teacher" if bool(args.use_teacher_loss) else "NoTeacher"

snapshot_path = (
    f"../models/{args.exp}/{args.model.upper()}_{args.labelnum}labels_"
    f"{args.consistency_type}{gamma_str}_{focal_str}_{teacher_str}_temp{args.temp}"
    f"{beta_str}_max_iterations{args.max_iterations}"
)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

batch_size = args.batch_size
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = True
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

num_classes = args.num_classes = 2
patch_size = tuple(args.patch_size)


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


def cleanup_checkpoints(snapshot_path, keep_top_n=5):
    """Delete all but the `keep_top_n` best iter_*_dice_*.pth checkpoints (score parsed from the file name)."""
    try:
        checkpoints = [f for f in os.listdir(snapshot_path) if f.startswith('iter_') and f.endswith('.pth')]
        if len(checkpoints) <= keep_top_n:
            return

        scores_and_files = []
        for f in checkpoints:
            try:
                # File name format: iter_{iter_num}_dice_{dice_score}.pth
                parts = f.replace('.pth', '').split('_')
                dice_index = parts.index('dice')
                score = float(parts[dice_index + 1])
                scores_and_files.append((score, f))
            except (ValueError, IndexError):
                continue

        if len(scores_and_files) <= keep_top_n:
            return

        scores_and_files.sort(key=lambda x: x[0], reverse=True)
        for score, filename in scores_and_files[keep_top_n:]:
            os.remove(os.path.join(snapshot_path, filename))
            logging.info(f"Deleted checkpoint with lower score ({score:.4f}): {filename}")
    except Exception as e:
        logging.warning(f"Could not clean up checkpoints: {e}")


if __name__ == "__main__":
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        use_pretrain = bool(args.use_pretrain) if args.model == 'swin_umamba' else False
        net = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=num_classes, scaler=args.feature_scaler,
                             chunk_size=args.chunk_size, use_pretrain=use_pretrain, pretrained_ckpt=args.pretrained_ckpt)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    # Keep a copy of the code next to the checkpoints.
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    ignore=shutil.ignore_patterns('.git', '__pycache__', 'models', 'data', '*.pth', '*.h5'))
    logging.info("Total params of model: {:.2f}M".format(sum(p.numel() for p in model.parameters())/1e6))

    db_train = Pancreas(base_dir=args.root_dir,
                        split='train',
                        transform=T.Compose([
                            RandomCrop(patch_size),
                            RandomRotFlip(),
                            ToTensor(),
                        ]))

    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, db_train.__len__()))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    # Fully supervised baseline when every scan in the batch is labeled,
    # otherwise each batch mixes labeled and unlabeled scans.
    if batch_size == labeled_bs:
        logging.info("Fully supervised mode (labeled scans only)")
        sampler = SubsetRandomSampler(labeled_idxs)
        trainloader = DataLoader(db_train, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True,
                                 worker_init_fn=worker_init_fn, drop_last=True)
    else:
        logging.info("Semi-supervised mode (labeled + unlabeled scans)")
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                                 worker_init_fn=worker_init_fn)

    # With VMamba initialization the pretrained encoder trains at a 10x lower learning rate.
    if args.model == 'swin_umamba' and args.use_pretrain:
        pretrained_params = []
        scratch_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name:
                pretrained_params.append(param)
            else:
                scratch_params.append(param)

        param_groups = [
            {'params': pretrained_params, 'lr': base_lr * 0.1},
            {'params': scratch_params, 'lr': base_lr}
        ]
    else:
        param_groups = model.parameters()

    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=0.01)
        logging.info(f'Using AdamW optimizer with lr={base_lr}')
    else:
        optimizer = optim.SGD(param_groups, lr=base_lr, momentum=0.9, weight_decay=0.0001)
        logging.info(f'Using SGD optimizer with lr={base_lr}')

    optimizer.zero_grad()

    iter_num = 0
    start_epoch = 0
    max_epoch = max_iterations // len(trainloader) + 1

    # Linear warm-up, then cosine decay (stepped once per epoch).
    warmup_epochs = min(args.warmup_epochs, max_epoch // 2)
    if warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch - warmup_epochs, eta_min=base_lr * 0.01)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=base_lr * 0.01)

    best_performance = 0.0

    # Resume from a full-state checkpoint (training was split across time-limited GPU sessions).
    if args.load_path and os.path.exists(args.load_path):
        logging.info(f'Loading checkpoint: {args.load_path}')
        checkpoint = torch.load(args.load_path, map_location='cuda:0', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'ema_model_state_dict' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'iter_num' in checkpoint:
                iter_num = checkpoint['iter_num']
                start_epoch = iter_num // len(trainloader)
                logging.info(f"Resuming from epoch {start_epoch}, iteration {iter_num}")
            if 'best_performance' in checkpoint:
                best_performance = checkpoint['best_performance']
        else:
            model.load_state_dict(checkpoint)

        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.copy_(param.data)

    model.train()
    ema_model.train()

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    else:
        consistency_criterion = losses.softmax_kl_loss

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} Iterations per epoch".format(len(trainloader)))

    iterator = tqdm(range(start_epoch, max_epoch), initial=start_epoch, total=max_epoch, ncols=110)

    use_amp = bool(args.use_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    uncl_criterion = dycon_losses.UnCLoss()
    fecl_criterion = dycon_losses.FeCLoss(device="cuda:0", temperature=args.temp, gamma=args.gamma,
                                          use_focal=bool(args.use_focal), rampup_epochs=1500)

    for epoch_num in iterator:

        if args.s_beta is not None:
            beta = args.s_beta
        else:
            beta = dycon_losses.adaptive_beta(epoch=epoch_num, total_epochs=max_epoch, max_beta=args.beta_max, min_beta=args.beta_min)

        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            if iter_num == 0:
                logging.info(f"Label shape: {tuple(label_batch.shape)}, unique labels: {torch.unique(label_batch).tolist()}")

            # The teacher sees a noise-perturbed copy of the batch.
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise

            with torch.autocast(device_type='cuda', enabled=use_amp):
                _, stud_logits, stud_features = model(volume_batch)
                with torch.no_grad():
                    _, ema_logits, ema_features = ema_model(ema_inputs)

                stud_probs = F.softmax(stud_logits.float(), dim=1)
                ema_probs = F.softmax(ema_logits.float(), dim=1)
                consistency_weight = get_current_consistency_weight(iter_num//150)

                # Supervised loss on the labeled scans. The pancreas is a tiny fraction of
                # each patch, so cross-entropy weights the foreground class 9x the background.
                class_weights = torch.tensor([0.1, 0.9], device='cuda')
                loss_seg = F.cross_entropy(stud_logits[:labeled_bs], label_batch[:labeled_bs], weight=class_weights)
                loss_seg_dice = losses.dice_loss(stud_probs[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)

                B, C, _, _, _ = stud_features.shape
                stud_embedding = stud_features.view(B, C, -1)
                stud_embedding = torch.transpose(stud_embedding, 1, 2)
                stud_embedding = F.normalize(stud_embedding, dim=-1)

                ema_embedding = ema_features.view(B, C, -1)
                ema_embedding = torch.transpose(ema_embedding, 1, 2)
                ema_embedding = F.normalize(ema_embedding, dim=-1)

                # Labels downsampled to the resolution of the contrastive features.
                mask_con = F.interpolate(label_batch.unsqueeze(1).float(), scale_factor=1/(args.feature_scaler * 4),
                                         mode='trilinear', align_corners=False).squeeze(1)
                mask_con = (mask_con > 0.5).float()
                mask_con = mask_con.reshape(B, -1)
                mask_con = mask_con.unsqueeze(1)

                # Unsupervised terms (FeCL, UnCL, consistency) apply only to unlabeled scans.
                if batch_size == labeled_bs:
                    f_loss = torch.tensor(0.0, device='cuda')
                    u_loss = torch.tensor(0.0, device='cuda')
                    consistency_loss = torch.tensor(0.0, device='cuda')
                else:
                    teacher_feat = ema_embedding[labeled_bs:] if args.use_teacher_loss else None
                    f_loss = fecl_criterion(feat=stud_embedding[labeled_bs:],
                                            mask=mask_con[labeled_bs:],
                                            teacher_feat=teacher_feat,
                                            gambling_uncertainty=None,
                                            epoch=epoch_num)
                    u_loss = uncl_criterion(stud_logits, ema_logits, beta)
                    consistency_loss = consistency_criterion(stud_probs[labeled_bs:], ema_probs[labeled_bs:]).mean()

                loss = args.l_weight * (loss_seg + loss_seg_dice) + consistency_weight * consistency_loss + args.u_weight * (f_loss + u_loss)
                loss = loss / args.accumulation_steps

            # Skip the step if the loss is not finite.
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"NaN or Inf found in loss at iteration {iter_num}")
                del noise, stud_embedding, ema_logits, ema_features, ema_probs, mask_con
                torch.cuda.empty_cache()
                continue

            scaler.scale(loss).backward()

            if (iter_num + 1) % args.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            with torch.no_grad():
                outputs_bin = (stud_probs[:, 1, :, :, :] > 0.5).float()
                dice_score = metrics.compute_dice(outputs_bin, label_batch)

            current_loss = (loss * args.accumulation_steps).item()
            current_dice = dice_score.mean().item()
            iterator.set_postfix(Loss=f"{current_loss:.4f}", Dice=f"{current_dice:.4f}")
            writer.add_scalar('train/loss', current_loss, iter_num)
            writer.add_scalar('train/dice', current_dice, iter_num)
            writer.add_scalar('train/lr', optimizer.param_groups[-1]['lr'], iter_num)

            # Every 200 iterations: validation Dice, best checkpoint and pruning.
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_3d_patch.var_all_case_Pancreas(model, args.root_dir, num_classes=args.num_classes,
                                                                 patch_size=patch_size, stride_xy=64, stride_z=64,
                                                                 list_name=args.val_list)
                writer.add_scalar('val/dice', avg_metric, iter_num)
                if avg_metric > best_performance:
                    best_performance = round(float(avg_metric), 4)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    state = {'iter_num': iter_num, 'model_state_dict': model.state_dict(), 'ema_model_state_dict': ema_model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                             'best_performance': best_performance}
                    torch.save(state, save_mode_path)
                    torch.save(state, save_best)
                    cleanup_checkpoints(snapshot_path, keep_top_n=args.keep_top_k)
                logging.info('Iteration %d : Dice: %03f Best_dice: %03f' % (iter_num, avg_metric, best_performance))
                model.train()

            # Every 100 iterations: full-state checkpoint for resuming.
            if iter_num > 0 and iter_num % 100 == 0:
                save_latest_path = os.path.join(snapshot_path, 'latest_model.pth')
                torch.save({'iter_num': iter_num, 'model_state_dict': model.state_dict(), 'ema_model_state_dict': ema_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                            'best_performance': best_performance}, save_latest_path)

            if iter_num >= max_iterations:
                break

        scheduler.step()

        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    logging.info("Training finished")

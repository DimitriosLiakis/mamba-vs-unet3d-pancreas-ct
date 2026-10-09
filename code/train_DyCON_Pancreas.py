import os
import sys
import shutil
import random
import logging
import argparse
import numpy as np

# Fix for Numpy 2.0 compatibility with Medpy
if not hasattr(np, 'bool'): np.bool = np.bool_
if not hasattr(np, 'float_'): np.float_ = np.float64

from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms as T

from networks.net_factory_3d import net_factory_3d
from utils import ramps, metrics, losses, dycon_losses, test_3d_patch, monitor
from dataloaders.pancreas import Pancreas, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler 

parser = argparse.ArgumentParser(description="Training DyCON on PancreasCT Dataset")

parser.add_argument('--root_dir', type=str, default="../data/Pancreas", help='Path to PancreasCT dataset')
parser.add_argument('--patch_size', type=str, default=[64, 64, 64], help='Input image patch size')
parser.add_argument('--exp', type=str, default='PancreasCT', help='Experiment name')
parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training') # DDP
parser.add_argument('--seed', type=int, default=1337, help='Random seed for reproducibility')
parser.add_argument('--deterministic', type=int, default=1, help='Use deterministic training (0 or 1)')

parser.add_argument('--model', type=str, choices=['unet_3D', 'vnet', 'swin_umamba'], default='unet_3D', help='Model architecture')
parser.add_argument('--in_ch', type=int, default=1, help='Number of input channels')
parser.add_argument('--num_classes', type=int, default=2, help='Number of segmentation classes')
parser.add_argument('--feature_scaler', type=int, default=2, help='Feature scaling factor for contrastive loss')

parser.add_argument('--max_iterations', type=int, default=10000, help='Maximum number of training iterations')
parser.add_argument('--batch_size', type=int, default=8, help='Total batch size per GPU')
parser.add_argument('--labeled_bs', type=int, default=4, help='Labeled batch size per GPU')
parser.add_argument('--base_lr', type=float, default=0.01, help='Base learning rate')

parser.add_argument('--labelnum', type=int, default=12, help='Number of labeled samples per class')
parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay for teacher model')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_type', type=str, default="mse", help='Consistency loss type')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='Ramp-up duration for consistency weight')

parser.add_argument('--gamma', type=float, default=2.0)
parser.add_argument('--beta_min', type=float, default=0.5)
parser.add_argument('--beta_max', type=float, default=5.0)
parser.add_argument('--s_beta', type=float, default=None)
parser.add_argument('--temp', type=float, default=0.6)
parser.add_argument('--l_weight', type=float, default=1.0)
parser.add_argument('--u_weight', type=float, default=0.5)
parser.add_argument('--use_focal', type=int, default=1)
parser.add_argument('--use_teacher_loss', type=int, default=1)

parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'])
parser.add_argument('--use_amp', type=int, default=0)
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--chunk_size', type=int, default=16)
parser.add_argument('--warmup_epochs', type=int, default=100)
parser.add_argument('--accumulation_steps', type=int, default=2)
# ΝΕΟ ARGUMENT για καθαρισμό checkpoints
parser.add_argument('--keep_top_k', type=int, default=5, help='Number of best checkpoints to keep, to save disk space.')
# ΝΕΟ ARGUMENT για προεκπαιδευμένα βάρη
parser.add_argument('--use_pretrain', type=int, default=0, help='Use pretrained VMamba/ImageNet weights for SwinUMamba')
parser.add_argument('--pretrained_ckpt', type=str, default='./data/pretrained/vmamba/vmamba_tiny_e292.pth', help='Path to pretrained VMamba weights')

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

# --- DDP SETUP ---
# Διαβάζουμε το LOCAL_RANK από το περιβάλλον του torchrun
if "LOCAL_RANK" in os.environ:
    args.local_rank = int(os.environ["LOCAL_RANK"])

if args.local_rank != -1:
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    is_master = local_rank == 0
else:
    # Allow running without DDP
    local_rank = 0
    is_master = True

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

# Ensure different random seeds for each process
random.seed(args.seed + local_rank)
np.random.seed(args.seed + local_rank)
torch.manual_seed(args.seed + local_rank)
torch.cuda.manual_seed(args.seed + local_rank)

num_classes = args.num_classes = 2
patch_size = args.patch_size = (64, 64, 64) 

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # Use model.module parameters in DDP
    student_params = model.module.parameters() if isinstance(model, DDP) else model.parameters()
    for ema_param, param in zip(ema_model.parameters(), student_params):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def cleanup_checkpoints(snapshot_path, keep_top_n=5):
    """Deletes old best checkpoints, keeping only the top N based on dice score in filename."""
    if not is_master:
        return
    try:
        # Βρίσκει όλα τα checkpoints που έχουν αποθηκευτεί με βάση την απόδοσή τους
        checkpoints = [f for f in os.listdir(snapshot_path) if f.startswith('iter_') and f.endswith('.pth')]
        
        # Αν έχουμε λιγότερα ή ίσα checkpoints από όσα θέλουμε να κρατήσουμε, δεν κάνει τίποτα
        if len(checkpoints) <= keep_top_n:
            return

        # "Διαβάζει" το σκορ (dice) από το όνομα του κάθε αρχείου
        scores_and_files = []
        for f in checkpoints:
            try:
                # Format ονόματος: iter_{iter_num}_dice_{dice_score}.pth
                parts = f.replace('.pth', '').split('_')
                dice_index = parts.index('dice')
                score = float(parts[dice_index + 1])
                scores_and_files.append((score, f))
            except (ValueError, IndexError):
                # Αγνοεί αρχεία που δεν ταιριάζουν στο format
                continue
        
        if len(scores_and_files) <= keep_top_n:
            return

        # Ταξινομεί τα αρχεία από το καλύτερο (μεγαλύτερο dice) στο χειρότερο
        scores_and_files.sort(key=lambda x: x[0], reverse=True)
        
        # Αρχεία προς διαγραφή (όλα εκτός από τα N καλύτερα)
        to_delete = scores_and_files[keep_top_n:]
        
        for score, filename in to_delete:
            file_path = os.path.join(snapshot_path, filename)
            os.remove(file_path)
            logging.info(f"🗑️ Deleted checkpoint with lower score ({score:.4f}): {filename}")
    except Exception as e:
        logging.warning(f"Could not clean up checkpoints: {e}")

if __name__ == "__main__":
    if is_master and not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    def create_model(ema=False):
        # Προώθηση use_pretrain μόνο για swin_umamba
        use_pretrain = bool(args.use_pretrain) if args.model == 'swin_umamba' else False
        net = net_factory_3d(net_type=args.model, in_chns=args.in_ch, class_num=num_classes, scaler=args.feature_scaler, chunk_size=args.chunk_size, use_pretrain=use_pretrain, pretrained_ckpt=args.pretrained_ckpt)        
        model = net.cuda(local_rank)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    if args.local_rank != -1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    ema_model = create_model(ema=True)
    
    if is_master:
        # Copy code only once from the master process
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

    # ==========================================
    # ΕΞΥΠΝΟΣ ΦΟΡΤΩΤΗΣ ΔΕΔΟΜΕΝΩΝ (PURE SUPERVISED VS SSL)
    # ==========================================
    train_sampler = None
    if batch_size == labeled_bs:
        if is_master: logging.info("🔥 PURE SUPERVISED MODE ACTIVATED 🔥")
        if args.local_rank != -1:
            train_sampler = DistributedSampler(labeled_idxs)
        else:
            from torch.utils.data.sampler import SubsetRandomSampler
            train_sampler = SubsetRandomSampler(labeled_idxs)
        trainloader = DataLoader(db_train, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
    else:
        if is_master: logging.info("🔥 SEMI-SUPERVISED MODE ACTIVATED 🔥")
        batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)
        # Note: TwoStreamBatchSampler is not DDP-aware. Each process will sample randomly.
        trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
        
    # ==========================================
    # Differential Learning Rates (για Pretrain)
    # ==========================================
    if args.model == 'swin_umamba' and args.use_pretrain:
        pretrained_params = []
        scratch_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name: # Προεκπαιδευμένα βάρη Mamba
                pretrained_params.append(param)
            else:
                scratch_params.append(param)
        
        param_groups = [
            {'params': pretrained_params, 'lr': base_lr * 0.1}, # 10x μικρότερο LR στον encoder!
            {'params': scratch_params, 'lr': base_lr}
        ]
    else:
        param_groups = model.parameters()

    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=0.01)
        if is_master: logging.info(f'Using AdamW optimizer with lr={base_lr}')
    else:
        optimizer = optim.SGD(param_groups, lr=base_lr, momentum=0.9, weight_decay=0.0001)
        if is_master: logging.info(f'Using SGD optimizer with lr={base_lr}')

    optimizer.zero_grad()

    iter_num = 0
    start_epoch = 0
    max_epoch = max_iterations // len(trainloader) + 1
    
    warmup_epochs = min(args.warmup_epochs, max_epoch // 2)
    if warmup_epochs > 0:
        warmup_sched = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch - warmup_epochs, eta_min=base_lr * 0.01)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_sched, cosine_sched], milestones=[warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=base_lr * 0.01)

    best_performance = 0.0

    # ==========================================
    # 🔥 ΝΕΟ ΚΑΙ ΔΙΟΡΘΩΜΕΝΟ LOADING MECHANISM 🔥
    # ==========================================
    if args.load_path and os.path.exists(args.load_path):
        if is_master: logging.info(f'\n🚀 LOADING CHECKPOINT: {args.load_path}\n')
        checkpoint = torch.load(args.load_path, map_location=f'cuda:{local_rank}')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            if 'ema_model_state_dict' in checkpoint:
                ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'iter_num' in checkpoint:
                iter_num = checkpoint['iter_num']
                start_epoch = iter_num // len(trainloader) # Recalculate start epoch
                if is_master: logging.info(f"✅ Resuming exactly from Epoch {start_epoch}, Iteration {iter_num}")
            if 'best_performance' in checkpoint:
                best_performance = checkpoint['best_performance']
        else:
            model_to_load = model.module if isinstance(model, DDP) else model
            model_to_load.load_state_dict(checkpoint)
            
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.copy_(param.data)

    model.train()
    ema_model.train()
    
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log') if is_master else None
    if is_master: logging.info("{} Iterations per epoch".format(len(trainloader)))

    # Ξεκινάμε την μπάρα από τη σωστή Εποχή!
    iterator = tqdm(range(start_epoch, max_epoch), initial=start_epoch, total=max_epoch, ncols=110, disable=not is_master)
    # Set epoch for DistributedSampler to ensure proper shuffling
    if args.local_rank != -1 and train_sampler is not None:
        train_sampler.set_epoch(start_epoch)

    use_amp = bool(args.use_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    uncl_criterion = dycon_losses.UnCLoss()
    fecl_criterion = dycon_losses.FeCLoss(device=f"cuda:{local_rank}", temperature=args.temp, gamma=args.gamma, use_focal=bool(args.use_focal), rampup_epochs=1500)
    
    for epoch_num in iterator:

        if args.s_beta is not None:
            beta = args.s_beta
        else:
            beta = dycon_losses.adaptive_beta(epoch=epoch_num, total_epochs=max_epoch, max_beta=args.beta_max, min_beta=args.beta_min)

        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(local_rank), sampled_batch['label'].cuda(local_rank)
            
            # --- ΤΑ DIAGNOSTICS ΜΑΣ (Μόνο στην αρχή) ---
            if iter_num == 0 and is_master:
                print(f"\n{'='*60}")
                print(f"🔥 DIAGNOSTICS: Label Shape: {label_batch.shape}")
                print(f"🔥 DIAGNOSTICS: Unique Labels: {torch.unique(label_batch)}")
                print(f"{'='*60}\n")
            
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = volume_batch + noise

            # 🔥 ΔΙΟΡΘΩΣΗ: Όλο το Forward Pass & Loss μέσα στο autocast
            with torch.autocast(device_type='cuda', enabled=use_amp):
                _, stud_logits, stud_features = model(volume_batch) 
                with torch.no_grad():
                    _, ema_logits, ema_features = ema_model(ema_inputs)
                
                stud_probs = F.softmax(stud_logits.float(), dim=1)
                ema_probs = F.softmax(ema_logits.float(), dim=1)
                consistency_weight = get_current_consistency_weight(iter_num//150)
                
                # ==========================================
                # 🔥 Η ΛΥΣΗ ΓΙΑ ΤΟ CLASS IMBALANCE 🔥
                # ==========================================
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
                
                mask_con = F.interpolate(label_batch.unsqueeze(1).float(), scale_factor=1/(args.feature_scaler * 4), mode='trilinear', align_corners=False).squeeze(1)  
                mask_con = (mask_con > 0.5).float()
                mask_con = mask_con.reshape(B, -1)
                mask_con = mask_con.unsqueeze(1) 

                # ΠΡΟΣΤΑΣΙΑ ΝΑΝ & SUPERVISED MODE
                if batch_size == labeled_bs:
                    f_loss = torch.tensor(0.0, device='cuda')
                    u_loss = torch.tensor(0.0, device='cuda')
                    consistency_loss = torch.tensor(0.0, device='cuda')
                else:
                    teacher_feat = ema_embedding if args.use_teacher_loss else None
                    f_loss = fecl_criterion(feat=stud_embedding[labeled_bs:],
                                            mask=mask_con[labeled_bs:], 
                                            teacher_feat=teacher_feat[labeled_bs:], 
                                            gambling_uncertainty=None, 
                                            epoch=epoch_num)
                    u_loss = uncl_criterion(stud_logits, ema_logits, beta)
                    consistency_loss = consistency_criterion(stud_probs[labeled_bs:], ema_probs[labeled_bs:]).mean()
                
                loss = args.l_weight * (loss_seg + loss_seg_dice) + consistency_weight * consistency_loss + args.u_weight * (f_loss + u_loss)
                loss = loss / args.accumulation_steps

            if torch.isnan(loss) or torch.isinf(loss):
                if is_master: logging.warning(f"NaN or Inf found in loss at iteration {iter_num}")
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
                H, W, D = stud_logits.shape[-3:]
                max_dist = np.linalg.norm([H, W, D])
                hausdorff_score = metrics.compute_hd95(outputs_bin, label_batch, max_dist)

            current_loss = (loss * args.accumulation_steps).item()
            current_dice = dice_score.mean().item()
            if is_master:
                iterator.set_postfix(Loss=f"{current_loss:.4f}", Dice=f"{current_dice:.4f}")

            # ==========================================
            # 🔥 EVALUATION & SAVING 🔥
            # ==========================================
            if iter_num > 0 and iter_num % 200 == 0 and is_master: 
                model.eval()
                model_to_test = model.module if isinstance(model, DDP) else model
                avg_metric = test_3d_patch.var_all_case_Pancreas(model_to_test, args.root_dir, num_classes=args.num_classes, patch_size=patch_size, stride_xy=64, stride_z=64)
                if avg_metric > best_performance:
                    best_performance = round(avg_metric, 4)
                    save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format( iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    state_dict_to_save = model_to_test.state_dict()
                    torch.save({'iter_num': iter_num, 'model_state_dict': state_dict_to_save, 'ema_model_state_dict': ema_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_performance': best_performance},  save_mode_path)
                    torch.save({'iter_num': iter_num, 'model_state_dict': state_dict_to_save, 'ema_model_state_dict': ema_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_performance': best_performance},  save_best)
                    
                    # 🔥 ΚΑΘΑΡΙΣΜΟΣ ΠΑΛΙΩΝ CHECKPOINTS 🔥
                    cleanup_checkpoints(snapshot_path, keep_top_n=args.keep_top_k)
                logging.info('Iteration %d : Dice: %03f Best_dice: %03f' % (iter_num, avg_metric, best_performance))
                model.train()
                
            if iter_num > 0 and iter_num % 100 == 0 and is_master:
                save_latest_path = os.path.join(snapshot_path, 'latest_model.pth')
                model_to_test = model.module if isinstance(model, DDP) else model
                state_dict_to_save = model_to_test.state_dict()
                torch.save({'iter_num': iter_num, 'model_state_dict': state_dict_to_save, 'ema_model_state_dict': ema_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_performance': best_performance},  save_latest_path)

            if iter_num >= max_iterations:
                break

        scheduler.step()

        if iter_num >= max_iterations:
            if is_master: iterator.close()
            break
            
    if is_master:
        writer.close()
        print("Training Finished!")

    # Clean up DDP
    if args.local_rank != -1:
        dist.destroy_process_group()
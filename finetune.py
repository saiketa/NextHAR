import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary as tsummary

from config import get_args
from utils.dataloader import IMUDataset_fft
from utils.util import split_data_label, change_windowsize
from models.pretrain import PretrainTransformer
from models.finetune import FineTuneModel


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(args):
    data = np.load(os.path.join(args.data_dir, args.dataset, args.data_file)).astype(np.float32)
    labels = np.load(os.path.join(args.data_dir, args.dataset, args.label_file)).astype(np.float32)
    data, labels = change_windowsize(data, labels, args.L)
    data_train, label_train, data_vali, label_vali, data_test, label_test = split_data_label(
        data, labels, args.train_rate, args.valid_rate
    )
    total = len(data_train)
    k = max(1, int(args.ft_frac * total))
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(total, size=k, replace=False)
    ft_data, ft_label = data_train[idx], label_train[idx]
    ds_tr = IMUDataset_fft(ft_data, ft_label, isNormalization=args.normalization)
    ds_va = IMUDataset_fft(data_vali, label_vali, isNormalization=args.normalization)
    ds_te = IMUDataset_fft(data_test, label_test, isNormalization=args.normalization)
    ld_tr = DataLoader(ds_tr, shuffle=True, batch_size=args.finetune_batch_size, drop_last=False)
    ld_va = DataLoader(ds_va, shuffle=False, batch_size=args.finetune_batch_size, drop_last=False)
    ld_te = DataLoader(ds_te, shuffle=False, batch_size=args.finetune_batch_size, drop_last=False)
    return ld_tr, ld_va, ld_te


@torch.no_grad()
def evaluate_loss(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    for x_orgin, _, y in loader:
        x_orgin = x_orgin.cuda()
        y = y.cuda().long()
        logits = model(x_orgin)
        loss = criterion(logits, y)
        total += loss.item()
        n += 1
    return total / max(1, n)


def main():
    args = get_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    set_seed(args.seed)
    ld_tr, ld_va, ld_te = build_loaders(args)

    ckpt_pre = os.path.join(args.model_path, f"{args.dataset}_pretrain.pth")
    if not os.path.exists(ckpt_pre):
        raise FileNotFoundError(ckpt_pre)

    pretrain = PretrainTransformer(
        in_channels=args.N,
        seq_length=args.L,
        patch_size=args.patch_size,
        stride=args.stride,
        embed_dim=args.embed_dim,
        num_layers=args.pretrain_layers,
        nhead=args.pretrain_nhead,
    ).cuda()
    pretrain.load_state_dict(torch.load(ckpt_pre))

    ft_model = FineTuneModel(pretrain, n_class=args.n_class).cuda()
    if args.ft_freeze:
        for name, p in ft_model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

    tsummary(ft_model, input_size=(1, args.N, args.L), col_names=("input_size", "output_size", "num_params"))

    params = [p for p in ft_model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=args.ft_lr)
    criterion = nn.CrossEntropyLoss().cuda()

    best_val, best_state = float("inf"), None
    save_path = os.path.join(args.model_path, f"{args.dataset}_finetune.pth")

    for epoch in range(args.ft_epochs):
        ft_model.train()
        total, n = 0.0, 0
        for x_orgin, _, y in ld_tr:
            x_orgin = x_orgin.cuda()
            y = y.cuda().long()
            opt.zero_grad()
            logits = ft_model(x_orgin)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        train_loss = total / max(1, n)
        val_loss = evaluate_loss(ft_model, ld_va, criterion)
        print(f"Epoch {epoch} | train {train_loss:.6f} | val {val_loss:.6f}")
        if val_loss < best_val:
            best_val, best_state = val_loss, ft_model.state_dict()
            torch.save(best_state, save_path)

    ft_model.load_state_dict(torch.load(save_path))
    test_loss = evaluate_loss(ft_model, ld_te, criterion)
    print(f"Best val {best_val:.6f} | test {test_loss:.6f}")


if __name__ == "__main__":
    main()
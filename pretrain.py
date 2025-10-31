import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary as tsummary
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from config import get_args
from utils.dataloader import IMUDataset_fft
from utils.util import split_data_label, change_windowsize
from models.pretrain import PretrainTransformer


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
    ds_tr = IMUDataset_fft(data_train, label_train, isNormalization=args.normalization)
    ds_va = IMUDataset_fft(data_vali, label_vali, isNormalization=args.normalization)
    ds_te = IMUDataset_fft(data_test, label_test, isNormalization=args.normalization)
    ld_tr = DataLoader(ds_tr, shuffle=True, batch_size=args.pretrain_batch_size, drop_last=False)
    ld_va = DataLoader(ds_va, shuffle=False, batch_size=args.pretrain_batch_size, drop_last=False)
    ld_te = DataLoader(ds_te, shuffle=False, batch_size=args.pretrain_batch_size, drop_last=False)
    return ld_tr, ld_va, ld_te


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total, n = 0.0, 0
    for x_orgin, _, _ in loader:
        x_orgin = x_orgin.cuda()
        optimizer.zero_grad()
        pred_next, target_next = model(x_orgin, return_targets=True)
        loss = criterion(pred_next, target_next)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(1, n)


def evaluate(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x_orgin, _, _ in loader:
            x_orgin = x_orgin.cuda()
            pred_next, target_next = model(x_orgin, return_targets=True)
            loss = criterion(pred_next, target_next)
            total += loss.item()
            n += 1
    return total / max(1, n)


def main():
    args = get_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    set_seed(args.seed)
    ld_tr, ld_va, ld_te = build_loaders(args)
    model = PretrainTransformer(
        in_channels=args.N,
        seq_length=args.L,
        patch_size=args.patch_size,
        stride=args.stride,
        embed_dim=args.embed_dim,
        num_layers=args.pretrain_layers,
        nhead=args.pretrain_nhead,
    ).cuda()
    tsummary(model, input_size=(1, args.N, args.L), col_names=("input_size", "output_size", "num_params"))
    optimizer = optim.Adam(model.parameters(), lr=args.pretrain_lr)
    criterion = nn.MSELoss().cuda()
    best, best_state = float("inf"), None
    save_path = os.path.join(args.model_path, f"{args.dataset}_pretrain.pth")
    for epoch in range(args.epochs):
        tr = train_one_epoch(model, ld_tr, criterion, optimizer)
        va = evaluate(model, ld_va, criterion)
        print(f"Epoch {epoch} | train {tr:.6f} | val {va:.6f}")
        if va < best:
            best, best_state = va, copy.deepcopy(model.state_dict())
            torch.save(best_state, save_path)
    model.load_state_dict(torch.load(save_path))
    te = evaluate(model, ld_te, criterion)
    print(f"Best val {best:.6f} | test {te:.6f}")


if __name__ == "__main__":
    main()
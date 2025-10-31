import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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
    data = np.load(os.path.join(args.data_dir, args.target_dataset, args.data_file)).astype(np.float32)
    labels = np.load(os.path.join(args.data_dir, args.target_dataset, args.label_file)).astype(np.float32)
    data, labels = change_windowsize(data, labels, args.L)
    
    data_train, label_train, data_vali, label_vali, data_test, label_test = \
        split_data_label(data, labels, 0.0, 0.0)

    ds_te = IMUDataset_fft(data_test, label_test, isNormalization=args.normalization)
    ld_te = DataLoader(ds_te, shuffle=False, batch_size=args.finetune_batch_size, drop_last=False)
    return ld_te


def evaluate(ft_model, loader):
    ft_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_orgin, _, y in loader:
            x_orgin = x_orgin.cuda()
            logits = ft_model(x_orgin)
            pred = torch.argmax(logits, dim=1).cpu().tolist()
            y_true += y.tolist()
            y_pred += pred
    conf = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1m = f1_score(y_true, y_pred, average="macro")
    f1mi = f1_score(y_true, y_pred, average="micro")
    return conf, acc, prec, rec, f1m, f1mi


def main():
    args = get_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    set_seed(args.seed)
    ld_te = build_loaders(args)
    ckpt_ft = os.path.join(args.model_path, f"{args.dataset}_finetune.pth")
    if not os.path.exists(ckpt_ft):
        raise FileNotFoundError(ckpt_ft)
    pre_base = PretrainTransformer(
        in_channels=args.N,
        seq_length=args.L,
        patch_size=args.patch_size,
        stride=args.stride,
        embed_dim=args.embed_dim,
        num_layers=args.pretrain_layers,
        nhead=args.pretrain_nhead,
    ).cuda()
    ft_model = FineTuneModel(pre_base, n_class=args.n_class).cuda()
    ft_model.load_state_dict(torch.load(ckpt_ft))
    conf, acc, prec, rec, f1m, f1mi = evaluate(ft_model, ld_te)
    print(f"Cross-dataset: src={args.dataset} tgt={args.target_dataset}")
    print(f"Acc {acc:.4f} P {prec:.4f} R {rec:.4f} F1(macro) {f1m:.4f} F1(micro) {f1mi:.4f}")
    print(conf)


if __name__ == "__main__":
    main()
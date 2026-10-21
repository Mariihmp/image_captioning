"""Train one captioning model on precomputed features.

The checkpoint with the lowest validation loss is kept (not the last epoch),
and training stops early when validation loss stops improving.

    python -m captioning.train --model transformer --seed 0
    python -m captioning.train --model attention --seed 0
    python -m captioning.train --model lstm --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CaptionDataset, Vocabulary, load_captions, load_features
from .models import MODELS, build_model
from .utils import get_device, set_seed

DEFAULT_LR = {"lstm": 4e-4, "attention": 4e-4, "transformer": 1e-4}


def caption_loss(model, features, inputs, targets, loss_fn, alpha_c):
    logits, alphas = model.decode(model.encode(features), inputs)
    loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    if alphas is not None and alpha_c > 0:
        # "Doubly stochastic" attention term from Show, Attend and Tell:
        # encourages the model to look at every image region at some point.
        mask = targets.ne(0).unsqueeze(2).float()
        loss = loss + alpha_c * ((1.0 - (alphas * mask).sum(dim=1)) ** 2).mean()
    return loss


def run_epoch(model, loader, loss_fn, device, alpha_c, optimizer=None, grad_clip=5.0):
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for features, inputs, targets in loader:
            features, inputs, targets = features.to(device), inputs.to(device), targets.to(device)
            loss = caption_loss(model, features, inputs, targets, loss_fn, alpha_c)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item() * len(inputs)
            count += len(inputs)
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--out-dir", default=None, help="defaults to runs/<model>_seed<seed>")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5, help="stop after this many epochs without improvement")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None, help="default depends on the model")
    parser.add_argument("--min-freq", type=int, default=5, help="minimum word count to enter the vocabulary")
    parser.add_argument("--max-len", type=int, default=30, help="longest training caption, in words")
    parser.add_argument("--alpha-c", type=float, default=1.0, help="attention regulariser (attention model only)")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out_dir or f"runs/{args.model}_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    lr = args.lr or DEFAULT_LR[args.model]

    captions = load_captions(Path(args.data_dir) / "captions.txt")
    train_images, train_features = load_features(args.features_dir, "train")
    val_images, val_features = load_features(args.features_dir, "val")

    # Vocabulary comes from training captions only, so no test words leak in.
    vocab = Vocabulary.build([c for img in train_images for c in captions[img]], args.min_freq)
    vocab.save(out_dir / "vocab.json")
    print(f"Vocabulary: {len(vocab)} words (min count {args.min_freq})")

    train_loader = DataLoader(CaptionDataset(train_images, train_features, captions, vocab, args.max_len),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(CaptionDataset(val_images, val_features, captions, vocab, args.max_len),
                            batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(args.model, len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters, device {device}")

    history, best_loss, best_epoch = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, device, args.alpha_c, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, device, args.alpha_c)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | {time.time() - start:.0f}s")

        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save({"model": args.model, "state_dict": model.state_dict(), "vocab_size": len(vocab),
                        "args": vars(args), "epoch": epoch, "val_loss": val_loss}, out_dir / "best.pt")
        elif epoch - best_epoch >= args.patience:
            print(f"No improvement for {args.patience} epochs, stopping.")
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best epoch {best_epoch} (val loss {best_loss:.4f}) saved to {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

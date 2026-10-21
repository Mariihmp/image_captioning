"""Generate captions for a split with a trained checkpoint, then score them.

    python -m captioning.predict --run-dir runs/transformer_seed0 --beam-size 3

Writes <run-dir>/predictions_<split>_beam<k>.json (caption plus the
log-probability of every word) and metrics_<split>_beam<k>.json.
"""

import argparse
import json
from pathlib import Path

import torch

from .data import Vocabulary, load_captions, load_features
from .decoding import beam_search
from .metrics import score_captions
from .models import build_model
from .utils import get_device


def load_model(run_dir, device):
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location=device)
    model = build_model(checkpoint["model"], checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--features-dir", default="features")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--spice", action="store_true", help="also compute SPICE (slow, needs Java)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    run_dir = Path(args.run_dir)
    model, checkpoint = load_model(run_dir, device)
    vocab = Vocabulary.load(run_dir / "vocab.json")
    images, features = load_features(args.features_dir, args.split)

    predictions = []
    for i, image in enumerate(images):
        words, logprobs = beam_search(model, features[i].float().to(device), vocab, args.beam_size, args.max_len)
        predictions.append({"image": image, "caption": " ".join(words), "token_logprobs": logprobs})
        print(f"\r{i + 1}/{len(images)}", end="")
    print()

    tag = f"{args.split}_beam{args.beam_size}"
    (run_dir / f"predictions_{tag}.json").write_text(json.dumps(predictions, indent=1))

    references = load_captions(Path(args.data_dir) / "captions.txt")
    metrics = score_captions(references, {p["image"]: p["caption"] for p in predictions}, spice=args.spice)
    metrics.update(model=checkpoint["model"], seed=checkpoint["args"]["seed"], split=args.split,
                   beam_size=args.beam_size, best_epoch=checkpoint["epoch"])
    (run_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}" if isinstance(value, float) else f"{name}: {value}")


if __name__ == "__main__":
    main()

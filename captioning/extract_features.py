"""Run EfficientNet once over every image and save the features per split.

The feature extractor is frozen, so computing features once (instead of every
epoch) gives the same result and makes training much faster.

    python -m captioning.extract_features --data-dir data --out-dir features
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import load_splits
from .features import IMAGE_TRANSFORM, FeatureExtractor
from .utils import get_device


class ImageFolderList(Dataset):
    def __init__(self, image_dir, images):
        self.image_dir = Path(image_dir)
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.image_dir / self.images[i]).convert("RGB")
        return IMAGE_TRANSFORM(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data", help="folder with captions.txt, split lists and Images/")
    parser.add_argument("--image-dir", default=None, help="defaults to <data-dir>/Images")
    parser.add_argument("--out-dir", default="features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    image_dir = args.image_dir or Path(args.data_dir) / "Images"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extractor = FeatureExtractor().to(device)

    for split, images in load_splits(args.data_dir).items():
        loader = DataLoader(ImageFolderList(image_dir, images), batch_size=args.batch_size,
                            num_workers=args.num_workers)
        chunks = []
        for i, batch in enumerate(loader):
            chunks.append(extractor(batch.to(device)).half().cpu())
            print(f"\r{split}: {min((i + 1) * args.batch_size, len(images))}/{len(images)}", end="")
        features = torch.cat(chunks)
        torch.save({"images": images, "features": features}, out_dir / f"{split}.pt")
        print(f"\n{split}: saved {tuple(features.shape)} to {out_dir / f'{split}.pt'}")


if __name__ == "__main__":
    main()

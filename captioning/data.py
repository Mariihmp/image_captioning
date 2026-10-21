"""Caption cleaning, the official Flickr8k splits, the vocabulary and the training dataset."""

import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIAL_TOKENS = [PAD, START, END, UNK]


def clean_caption(text):
    """Lowercase and keep only words made of letters.

    The same cleaning is used for training captions, reference captions and
    generated captions, so no model gets an advantage at scoring time.
    """
    return re.sub(r"[^a-z\s]", " ", text.lower()).split()


def load_captions(captions_file):
    """Read captions.txt (Kaggle Flickr8k format) into {image_name: [caption, ...]}."""
    captions = {}
    with open(captions_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header: image,caption
        for image, caption in reader:
            captions.setdefault(image, []).append(caption.strip())
    return captions


def _read_list(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_splits(data_dir):
    """Return {"train": [...], "val": [...], "test": [...]} image names.

    train (6000) and test (1000) are the official Flickr8k lists. val is every
    remaining image (1091), so it never overlaps with train or test.
    """
    data_dir = Path(data_dir)
    captions = load_captions(data_dir / "captions.txt")
    train = _read_list(data_dir / "Flickr_8k.trainImages.txt")
    test = _read_list(data_dir / "Flickr_8k.testImages.txt")
    used = set(train) | set(test)
    val = sorted(image for image in captions if image not in used)

    assert not set(train) & set(test), "train and test splits overlap"
    missing = used - set(captions)
    assert not missing, f"{len(missing)} split images have no captions"
    return {"train": train, "val": val, "test": test}


class Vocabulary:
    def __init__(self, words):
        self.itos = SPECIAL_TOKENS + [w for w in words if w not in SPECIAL_TOKENS]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, captions, min_freq=5):
        """Build from training captions only, keeping words seen at least min_freq times."""
        counts = Counter(w for caption in captions for w in clean_caption(caption))
        words = sorted(w for w, n in counts.items() if n >= min_freq)
        return cls(words)

    def __len__(self):
        return len(self.itos)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    @property
    def start_id(self):
        return self.stoi[START]

    @property
    def end_id(self):
        return self.stoi[END]

    def encode(self, caption):
        unk = self.stoi[UNK]
        return [self.start_id] + [self.stoi.get(w, unk) for w in clean_caption(caption)] + [self.end_id]

    def decode(self, ids):
        words = []
        for i in ids:
            if i == self.end_id:
                break
            if i not in (self.pad_id, self.start_id):
                words.append(self.itos[i])
        return words

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos[len(SPECIAL_TOKENS):]))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))


def load_features(features_dir, split):
    """Load the image features saved by captioning.extract_features.

    Returns (image_names, features) where features has shape (N, 64, 1280).
    """
    saved = torch.load(Path(features_dir) / f"{split}.pt")
    return saved["images"], saved["features"]


class CaptionDataset(Dataset):
    """One item per (image, caption) pair, so each image appears once per caption."""

    def __init__(self, images, features, captions, vocab, max_len=30):
        self.features = features
        self.vocab = vocab
        self.seq_len = max_len + 2  # <start> + words + <end>
        self.items = []
        for idx, image in enumerate(images):
            for caption in captions[image]:
                self.items.append((idx, vocab.encode(caption)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        idx, ids = self.items[i]
        ids = ids[: self.seq_len - 1] + [self.vocab.end_id] if len(ids) > self.seq_len else ids
        ids = ids + [self.vocab.pad_id] * (self.seq_len - len(ids))
        ids = torch.tensor(ids, dtype=torch.long)
        return self.features[idx].float(), ids[:-1], ids[1:]

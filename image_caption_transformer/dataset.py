import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from config import Config


class CaptionDataset(Dataset):
    """
    Custom Dataset for Flickr8k data.
    """

    def __init__(self, caption_data: dict, config: Config, word2idx: dict):
        self.caption_data = caption_data
        self.config = config
        self.word2idx = word2idx
        self.images = list(caption_data.keys())
        self.captions = list(caption_data.values())

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                (config.image_size, config.image_size)),
            torchvision.transforms.ToTensor()
        ])

    def load_and_prepare_images(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transforms(image)

    def load_and_prepare_captions(self, captions: list):
        preprocessed_captions = []
        for caption in captions:
            tokens = caption.strip().lower().split()
            if len(tokens) > self.config.maxlen:
                tokens = tokens[:self.config.maxlen]

            input_tokens = tokens[:-1].copy()
            target_tokens = tokens[1:].copy()
            sample_size = len(input_tokens)
            padding_size = self.config.maxlen - sample_size

            if padding_size > 0:
                padding_vec = ["<pad>"] * padding_size
                input_tokens += padding_vec
                target_tokens += padding_vec

            input_tokens = [self.word2idx.get(
                token, self.word2idx["<pad>"]) for token in input_tokens]
            target_tokens = [self.word2idx.get(
                token, self.word2idx["<pad>"]) for token in target_tokens]

            tgt_padding_mask = torch.ones(
                [self.config.maxlen, ], dtype=torch.bool)
            tgt_padding_mask[:sample_size] = False

            preprocessed_captions.append([
                torch.tensor(
                    input_tokens[:self.config.maxlen], dtype=torch.long),
                torch.tensor(
                    target_tokens[:self.config.maxlen], dtype=torch.long),
                tgt_padding_mask
            ])
        return preprocessed_captions

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.load_and_prepare_images(self.images[index])
        y = self.load_and_prepare_captions(self.captions[index])
        return image, y

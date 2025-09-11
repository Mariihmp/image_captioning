# /image-captioning-transformer/config.py

import torch


class Config:
    maxlen = 27  # max sequence length
    image_size = 256  # image size for resizing
    batch_size = 256  # batch size per dataloader
    epochs = 20  # num epochs
    learning_rate = 1e-4  # learning rate for optimizer
    d_model = 512   # model's dimension
    n_heads_encoder = 1  # encoders num_heads for multihead attention
    n_heads_decoder = 2  # decoders num_heads for multihead attention
    path_to_txt = r"E:/project_img_caption/data/captions.txt"  # path to captions data
    images_path = r"E:/project_img_caption/data/Images"  # path to images folder
    vocab_size = None  # this will be updated in the training script


# Instantiate the configuration
config = Config()

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# visualize_models.py

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# --- 1. Import from your actual config and model files ---
from config import config, device  # Imports your config and device directly
from model import FeatureExtractor, Encoder, Decoder

print("Successfully imported configuration from config.py.")

# --- 2. Handle vocab_size for visualization ---
# Your config sets vocab_size to None, but nn.Embedding requires a specific size.
# We'll set a temporary value here just for the purpose of building the graph.
if config.vocab_size is None:
    config.vocab_size = 10000  # A temporary, reasonable default
    print(
        f"Temporarily setting vocab_size to {config.vocab_size} for visualization.")

# --- 3. Setup SummaryWriter ---
writer = SummaryWriter('runs/image_captioning_architecture')
print(f"Device set to: {device}")
print("TensorBoard writer initialized. Logs will be saved in 'runs/image_captioning_architecture'.")

# --- 4. Initialize Models ---
# The models are now initialized using your imported config.
feat_ext = FeatureExtractor().to(device)
encoder = Encoder(config).to(device)
decoder = Decoder(config).to(device)
print("Models initialized and moved to device.")

# --- 5. Create Dummy Input Tensors using config values ---
# These tensors simulate a real batch of data based on your config.
dummy_images = torch.randn(
    config.batch_size, 3, config.image_size, config.image_size).to(device)
dummy_captions = torch.randint(
    0, config.vocab_size, (config.batch_size, config.maxlen)).to(device)

# Create a subsequent mask for the decoder's self-attention
tgt_mask = nn.Transformer.generate_square_subsequent_mask(
    config.maxlen).to(device)
print("Dummy input tensors created using dimensions from your config file.")

# --- 6. Trace Models and Add to TensorBoard ---
with torch.no_grad():
    print("Tracing models...")

    # Trace FeatureExtractor
    writer.add_graph(feat_ext, dummy_images)
    print("✅ FeatureExtractor graph added to TensorBoard.")

    # Pass data through to get intermediate shapes
    features_out = feat_ext(dummy_images)

    # Trace Encoder
    writer.add_graph(encoder, features_out)
    print("✅ Encoder graph added to TensorBoard.")

    # Pass data through to get intermediate shapes
    encoder_out = encoder(features_out)

    # Trace Decoder
    writer.add_graph(decoder, (dummy_captions, encoder_out, tgt_mask))
    print("✅ Decoder graph added to TensorBoard.")

writer.close()

print("\n--- All Done! ---")
print("To view the graphs, run the following command in your terminal:")
print("tensorboard --logdir=runs")

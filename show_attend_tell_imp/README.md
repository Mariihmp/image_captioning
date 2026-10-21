# Show, Attend and Tell (reference code)

The code in this folder comes from the public tutorial
[sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning),
an implementation of Xu et al., 2015, *Show, Attend and Tell: Neural Image
Caption Generation with Visual Attention*. Check that repository for its licence
before reusing the code.

It is kept here for reference only. Its settings (`train.py`, `eval.py`) still
point to the tutorial's own data paths and to MS COCO, and it was not trained
on Flickr8k in this project.

The attention model used in the thesis experiments is `SoftAttentionLSTM` in
`captioning/models.py`, written for the shared pipeline so it uses the same
split, vocabulary, image features and metrics as the other models.

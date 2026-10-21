import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from captioning import predict, train
from captioning.data import CaptionDataset, Vocabulary, clean_caption, load_captions, load_splits
from captioning.decoding import beam_search
from captioning.features import FeatureExtractor
from captioning.metrics import score_captions
from captioning.models import MODELS, build_model

REPO = Path(__file__).resolve().parents[1]


def test_clean_caption_keeps_short_words_and_drops_punctuation():
    assert clean_caption('A dog , "running" in 2 parks .') == ["a", "dog", "running", "in", "parks"]


def test_official_splits_are_disjoint_and_complete():
    splits = load_splits(REPO / "data")
    assert [len(splits[s]) for s in ("train", "val", "test")] == [6000, 1091, 1000]
    train, val, test = (set(splits[s]) for s in ("train", "val", "test"))
    assert not (train & val or train & test or val & test)


def test_captions_with_commas_are_read_whole():
    captions = load_captions(REPO / "data" / "captions.txt")
    assert len(captions) == 8091
    assert all(len(c) == 5 for c in captions.values())
    assert "A dog shakes its head near the shore , a red ball next to it ." in captions["1012212859_01547e3f17.jpg"]


def test_vocabulary_round_trip(tmp_path):
    vocab = Vocabulary.build(["a dog runs", "a dog sits", "a cat"], min_freq=2)
    assert vocab.decode(vocab.encode("a dog flies")) == ["a", "dog", "<unk>"]
    vocab.save(tmp_path / "v.json")
    assert Vocabulary.load(tmp_path / "v.json").itos == vocab.itos


def test_dataset_pads_and_truncates():
    vocab = Vocabulary.build(["a dog runs"], min_freq=1)
    features = torch.zeros(1, 64, 1280, dtype=torch.float16)
    dataset = CaptionDataset(["img"], features, {"img": ["a dog runs", "a dog " * 20]}, vocab, max_len=5)
    assert len(dataset) == 2
    feats, inputs, targets = dataset[1]
    assert feats.dtype == torch.float32
    assert inputs.shape == targets.shape == (6,)
    assert targets[-1].item() == vocab.end_id


@pytest.mark.parametrize("name", sorted(MODELS))
def test_models_learn_a_tiny_batch(name):
    torch.manual_seed(0)
    vocab_size, features = 20, torch.randn(4, 64, 1280)
    tokens = torch.randint(4, vocab_size, (4, 8))
    tokens[:, 0] = 1
    model = build_model(name, vocab_size)
    logits, alphas = model.decode(model.encode(features), tokens[:, :-1])
    assert logits.shape == (4, 7, vocab_size)
    if alphas is not None:
        assert alphas.shape == (4, 7, 64)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    losses = []
    for _ in range(30):
        loss = train.caption_loss(model, features, tokens[:, :-1], tokens[:, 1:], loss_fn, alpha_c=1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


@pytest.mark.parametrize("beam_size", [1, 3])
def test_beam_search_returns_words_and_logprobs(beam_size):
    vocab = Vocabulary.build(["a dog runs on grass"], min_freq=1)
    model = build_model("transformer", len(vocab)).eval()
    words, logprobs = beam_search(model, torch.randn(64, 1280), vocab, beam_size=beam_size, max_len=6)
    assert len(words) == len(logprobs) <= 6
    assert all(lp <= 0 for lp in logprobs)
    assert not {"<pad>", "<start>", "<end>"} & set(words)


def test_feature_extractor_gives_64_regions():
    features = FeatureExtractor(pretrained=False)(torch.randn(2, 3, 256, 256))
    assert features.shape == (2, 64, 1280)


def test_scores_perfect_caption_highest():
    refs = {"a": ["a dog runs on the grass", "a brown dog running"], "b": ["two people on a beach"]}
    good = score_captions(refs, {"a": "A dog runs on the grass .", "b": "two people on a beach"})
    bad = score_captions(refs, {"a": "a cat sleeps", "b": "a red car"})
    assert good["BLEU-1"] > bad["BLEU-1"] and good["CIDEr"] > bad["CIDEr"]


def test_train_and_predict_end_to_end(tmp_path, monkeypatch):
    """Runs the real command-line scripts on a fake 3-image dataset."""
    data_dir, features_dir, run_dir = tmp_path / "data", tmp_path / "features", tmp_path / "run"
    data_dir.mkdir()
    features_dir.mkdir()
    lines = ["image,caption"]
    for image in ["tr.jpg", "va.jpg", "te.jpg"]:
        lines += [f"{image},a dog runs on the grass ."] * 5
    (data_dir / "captions.txt").write_text("\n".join(lines))
    (data_dir / "Flickr_8k.trainImages.txt").write_text("tr.jpg\n")
    (data_dir / "Flickr_8k.testImages.txt").write_text("te.jpg\n")
    for split, image in [("train", "tr.jpg"), ("val", "va.jpg"), ("test", "te.jpg")]:
        torch.save({"images": [image], "features": torch.randn(1, 64, 1280).half()}, features_dir / f"{split}.pt")

    common = ["--data-dir", str(data_dir), "--features-dir", str(features_dir)]
    monkeypatch.setattr(sys, "argv", ["train", "--model", "lstm", "--epochs", "2", "--min-freq", "1",
                                      "--out-dir", str(run_dir), *common])
    train.main()
    assert (run_dir / "best.pt").exists() and (run_dir / "history.json").exists()

    monkeypatch.setattr(sys, "argv", ["predict", "--run-dir", str(run_dir), "--beam-size", "2", *common])
    predict.main()
    metrics = json.loads((run_dir / "metrics_test_beam2.json").read_text())
    assert metrics["model"] == "lstm" and "CIDEr" in metrics

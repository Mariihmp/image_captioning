"""Caption generation with beam search (beam_size=1 is greedy search)."""

import torch


@torch.no_grad()
def beam_search(model, features, vocab, beam_size=3, max_len=30):
    """Generate a caption for one image.

    features: (64, 1280) tensor for a single image.
    Returns (words, token_logprobs): the chosen words and the log-probability
    the model gave each of them (useful later for confidence analysis).
    The highest total log-probability wins, as in the Show, Attend and Tell tutorial code.
    """
    memory = model.encode(features.unsqueeze(0))
    device = features.device
    banned = [vocab.pad_id, vocab.start_id]

    beams = [([vocab.start_id], [], 0.0)]  # (token ids, token log-probs, total score)
    finished = []
    for _ in range(max_len + 1):
        tokens = torch.tensor([b[0] for b in beams], device=device)
        batch_memory = memory.repeat(len(beams), *[1] * (memory.dim() - 1))
        logits, _ = model.decode(batch_memory, tokens)
        logprobs = logits[:, -1].log_softmax(dim=-1)
        logprobs[:, banned] = float("-inf")

        # Each finished caption takes one place out of the beam.
        keep = beam_size - len(finished)
        top_values, top_ids = logprobs.topk(keep, dim=-1)
        candidates = []
        for b, (ids, lps, score) in enumerate(beams):
            for value, idx in zip(top_values[b].tolist(), top_ids[b].tolist()):
                candidates.append((ids + [idx], lps + [value], score + value))
        candidates.sort(key=lambda c: c[2], reverse=True)

        beams = []
        for candidate in candidates[:keep]:
            (finished if candidate[0][-1] == vocab.end_id else beams).append(candidate)
        if not beams:
            break
    finished.extend(beams)  # beams that hit max_len without <end>

    ids, lps, _ = max(finished, key=lambda c: c[2])
    words = vocab.decode(ids)[:max_len]
    return words, lps[: len(words)]

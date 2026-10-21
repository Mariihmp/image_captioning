"""Standard captioning metrics from the COCO caption toolkit (pycocoevalcap).

BLEU, ROUGE-L and CIDEr are pure Python. METEOR and SPICE need Java; they are
skipped with a warning when Java is not installed.
"""

import shutil
import warnings

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

from .data import clean_caption


def score_captions(references, predictions, spice=False):
    """references: {image: [caption, ...]}, predictions: {image: caption}. Returns {metric: score}.

    Both sides are cleaned with clean_caption so every model is scored the same way.
    """
    gts = {img: [" ".join(clean_caption(c)) for c in references[img]] for img in predictions}
    res = {img: [" ".join(clean_caption(predictions[img]))] for img in predictions}

    scorers = [(Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]), (Rouge(), "ROUGE-L"), (Cider(), "CIDEr")]
    if shutil.which("java"):
        from pycocoevalcap.meteor.meteor import Meteor
        scorers.append((Meteor(), "METEOR"))
        if spice:
            from pycocoevalcap.spice.spice import Spice
            scorers.append((Spice(), "SPICE"))
    else:
        warnings.warn("Java not found: METEOR and SPICE are skipped.")

    results = {}
    for scorer, names in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(names, list):
            results.update(zip(names, score))
        else:
            results[names] = score
    return {name: float(value) for name, value in results.items()}

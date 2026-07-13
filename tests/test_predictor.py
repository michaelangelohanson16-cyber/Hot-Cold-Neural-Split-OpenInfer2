"""
Predictor learnability: on synthetic, linearly separable activation data,
the low-rank predictor trained through PredictorTrainer's own code path
must recover the activation pattern (high F1) -- and its save/load
round-trip must preserve behavior. This validates the prefetch oracle's
training loop and serialization, not its accuracy on any real model.
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictor import NeuronPredictor, PredictorTrainer  # noqa: E402

H, I, RANK, N = 256, 128, 32, 4000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_synthetic(seed=0):
    """Ground truth is itself low-rank so the rank-RANK predictor can
    represent it: mask = (W2 @ W1 @ h + b > 0)."""
    torch.manual_seed(seed)
    w1 = torch.randn(RANK, H) / H ** 0.5
    w2 = torch.randn(I, RANK) / RANK ** 0.5
    b = torch.randn(I) * 0.01
    hidden = torch.randn(N, H)
    logits = hidden @ w1.T @ w2.T + b
    mask = (logits > 0).float()
    return hidden, mask


def test_trainer_learns_synthetic_pattern():
    hidden, mask = _make_synthetic()
    base_rate = mask.mean().item()
    assert 0.3 < base_rate < 0.7, "synthetic data should be roughly balanced"

    with tempfile.TemporaryDirectory() as tmp:
        data_file = os.path.join(tmp, "layer_000_expert_000.pt")
        torch.save({"hidden": hidden, "mask": mask.bool()}, data_file)

        trainer = PredictorTrainer(tmp, os.path.join(tmp, "out"),
                                   rank=RANK, device=DEVICE)
        from pathlib import Path
        # 4000 samples / batch 256 ~= 16 steps per epoch; separable low-rank
        # data needs a few hundred optimizer steps to converge from scratch,
        # so 25 epochs (~400 steps), not a token 3-4.
        predictor, metrics = trainer.train_one(Path(data_file),
                                               epochs=25, batch_size=256)
        final = metrics[f"epoch_25"]
        print(f"  final: f1={final['f1']:.3f} precision={final['precision']:.3f} "
              f"recall={final['recall']:.3f} (base rate {base_rate:.2f})")
        assert final["f1"] > 0.85, \
            f"predictor failed to learn separable data: f1={final['f1']:.3f}"
        assert final["recall"] > 0.80, \
            "miss rate too high on separable data -- prefetch would stall"


def test_save_load_roundtrip_preserves_outputs():
    hidden, mask = _make_synthetic(seed=1)
    model = NeuronPredictor(H, I, RANK).to(DEVICE)
    x = hidden[:8].to(DEVICE)
    with torch.no_grad():
        before = model(x)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.pt")
        model.save(path)
        loaded = NeuronPredictor.load(path, device=DEVICE)
        with torch.no_grad():
            after = loaded(x)
    assert torch.allclose(before, after, atol=1e-6), \
        "save/load changed predictor outputs"


def test_predict_mask_thresholding():
    model = NeuronPredictor(H, I, RANK).to(DEVICE)
    x = torch.randn(4, H, device=DEVICE)
    with torch.no_grad():
        probs = model(x)
        mask = model.predict_mask(x, threshold=0.5)
    assert mask.dtype == torch.bool
    assert torch.equal(mask, probs >= 0.5)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

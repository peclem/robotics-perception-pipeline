"""
Tests for AppearanceExtractor and friends.

TestNullExtractor       : Null implementation always returns Nones
TestDINOv2WrapperLogic  : crop handling, bbox clamping, batched call
                          (uses a fake DINOv2 module — no weights)
TestDINOv2Live          : real DINOv2 on GPU. Marked integration.
"""

from __future__ import annotations

import sys
import types
from typing import List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Null
# ---------------------------------------------------------------------------

class TestNullExtractor:

    def test_dim_zero(self):
        from perception.appearance_extractor import NullAppearanceExtractor
        assert NullAppearanceExtractor().embedding_dim == 0

    def test_returns_none_per_bbox(self):
        from perception.appearance_extractor import NullAppearanceExtractor
        ext = NullAppearanceExtractor()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = ext.extract(img, [(0, 0, 10, 10), (5, 5, 50, 50)])
        assert out == [None, None]

    def test_empty_bbox_list(self):
        from perception.appearance_extractor import NullAppearanceExtractor
        assert NullAppearanceExtractor().extract(
            np.zeros((10, 10, 3), dtype=np.uint8), []
        ) == []


# ---------------------------------------------------------------------------
# DINOv2 wrapper logic — without loading real weights
# ---------------------------------------------------------------------------

class _FakeProcessor:
    """Stand-in for transformers.AutoImageProcessor."""
    def __call__(self, images, return_tensors="pt"):
        import torch
        # Pretend to resize/normalise — return one (3, 14, 14) tensor per crop.
        n = len(images)
        return {"pixel_values": torch.zeros((n, 3, 14, 14))}


class _FakeOutput:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class _FakeModel:
    """Stand-in for transformers.AutoModel — returns deterministic CLS tokens."""
    def __init__(self):
        self.config = types.SimpleNamespace(hidden_size=4)
        self._called = 0

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, pixel_values):
        import torch
        n = pixel_values.shape[0]
        # Distinguishable per-call output to let tests check
        # which embedding came from which crop.
        # Hidden state shape: (N, seq_len, D); CLS is index 0.
        rows = []
        for i in range(n):
            seed = (self._called * 100 + i + 1) * 0.1
            rows.append(torch.tensor([
                [seed, seed * 0.5, -seed, 1.0]   # CLS
            ] + [[0.0] * 4] * 10))
        self._called += 1
        return _FakeOutput(torch.stack(rows))


@pytest.fixture
def fake_dinov2(monkeypatch):
    """Patch transformers' loaders so DINOv2AppearanceExtractor doesn't fetch weights."""
    fake_tx = types.ModuleType("transformers")

    def _from_pretrained_proc(name, **kw):
        return _FakeProcessor()

    def _from_pretrained_model(name, **kw):
        return _FakeModel()

    fake_tx.AutoImageProcessor = types.SimpleNamespace(
        from_pretrained=_from_pretrained_proc,
    )
    fake_tx.AutoModel = types.SimpleNamespace(
        from_pretrained=_from_pretrained_model,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_tx)
    yield


class TestDINOv2WrapperLogic:

    def test_embeddings_normalised(self, fake_dinov2):
        from perception.appearance_extractor import DINOv2AppearanceExtractor
        ext = DINOv2AppearanceExtractor(device="cpu")
        img = (np.random.default_rng(0).integers(
            0, 255, size=(200, 200, 3), dtype=np.uint8
        ))
        emb = ext.extract(img, [(10, 10, 110, 110), (50, 50, 180, 180)])
        assert len(emb) == 2
        for e in emb:
            assert e is not None
            assert e.shape == (4,)
            np.testing.assert_allclose(np.linalg.norm(e), 1.0, atol=1e-5)

    def test_invalid_bbox_yields_none(self, fake_dinov2):
        from perception.appearance_extractor import DINOv2AppearanceExtractor
        ext = DINOv2AppearanceExtractor(device="cpu")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Zero-area bbox + out-of-bounds bbox — both should yield None.
        out = ext.extract(img, [(50, 50, 50, 50), (500, 500, 600, 600),
                                (10, 10, 80, 80)])
        assert out[0] is None
        assert out[1] is None
        assert out[2] is not None

    def test_clamps_bbox_to_image_bounds(self, fake_dinov2):
        from perception.appearance_extractor import DINOv2AppearanceExtractor
        ext = DINOv2AppearanceExtractor(device="cpu")
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        # Bbox extends past image — should clamp to (0,0,100,100).
        emb = ext.extract(img, [(-50, -50, 200, 200)])
        assert emb[0] is not None     # crop survived the clamp

    def test_empty_call_returns_empty(self, fake_dinov2):
        from perception.appearance_extractor import DINOv2AppearanceExtractor
        ext = DINOv2AppearanceExtractor(device="cpu")
        assert ext.extract(np.zeros((10, 10, 3), dtype=np.uint8), []) == []

    def test_lazy_load(self, fake_dinov2):
        from perception.appearance_extractor import DINOv2AppearanceExtractor
        ext = DINOv2AppearanceExtractor(device="cpu")
        # No model yet
        assert ext._model is None
        # Triggering embedding_dim loads
        d = ext.embedding_dim
        assert d == 4
        assert ext._model is not None


# ---------------------------------------------------------------------------
# Live DINOv2 — hardware integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestDINOv2Live:

    def test_real_dinov2_returns_normalised_embedding(self):
        """Pull real DINOv2-small weights, run on a synthetic crop."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for DINOv2 live test")

        from perception.appearance_extractor import DINOv2AppearanceExtractor
        ext = DINOv2AppearanceExtractor()
        ext.warmup()
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)

        emb = ext.extract(img, [(100, 100, 300, 300), (200, 200, 400, 400)])
        assert len(emb) == 2
        for e in emb:
            assert e is not None
            np.testing.assert_allclose(np.linalg.norm(e), 1.0, atol=1e-4)
        # Distinct crops → distinct embeddings (cosine sim < 1.0).
        sim = float(np.dot(emb[0], emb[1]))
        assert sim < 1.0

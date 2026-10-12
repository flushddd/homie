"""ONNXRuntime policy wrapper (HomieDeploy-compatible)."""

from __future__ import annotations

from typing import Optional

import numpy as np


class OnnxPolicy:
    """Single-input history → actions via onnxruntime."""

    def __init__(self, path: str, hist_dim: int, providers: Optional[list] = None):
        import onnxruntime as ort

        self.hist_dim = int(hist_dim)
        self.sess = ort.InferenceSession(
            path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self._hist: Optional[np.ndarray] = None
        inp = self.sess.get_inputs()[0]
        # Validate trailing dim if static
        shape = inp.shape
        if len(shape) >= 2 and isinstance(shape[-1], int) and shape[-1] != self.hist_dim:
            raise ValueError(
                f"ONNX input dim {shape[-1]} != configured hist_dim {self.hist_dim}"
            )

    def reset(self) -> None:
        self._hist = None

    def __call__(self, one_step: np.ndarray) -> np.ndarray:
        step = np.asarray(one_step, dtype=np.float32).reshape(-1)
        one = step.shape[0]
        if self._hist is None:
            if self.hist_dim % one != 0:
                raise ValueError(f"hist_dim {self.hist_dim} not divisible by one-step {one}")
            self._hist = np.tile(step, self.hist_dim // one)
        else:
            self._hist = np.concatenate([self._hist[one:], step], axis=0)
        x = self._hist.reshape(1, -1)
        out = self.sess.run(None, {self.input_name: x})[0]
        return np.asarray(out, dtype=np.float32).reshape(-1)

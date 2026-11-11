"""Differentiable damped least-squares position IK (batch PyTorch).

Uses geometric positional Jacobian ``J`` and ``torch.linalg.solve`` — no
explicit inverse, no NumPy. Gradients flow through ``q`` and ``J`` when they
require grad (imagination must supply predicted-state Jacobians).
"""

from __future__ import annotations

import torch


def dls_step(
  J: torch.Tensor,
  err: torch.Tensor,
  *,
  damping: float = 0.08,
  max_dq: float = 0.05,
) -> torch.Tensor:
  """One DLS position step.

  Args:
    J: ``(B, 3, n)`` positional Jacobian.
    err: ``(B, 3)`` desired Cartesian step (same frame as J).
  Returns:
    ``dq`` ``(B, n)``.
  """
  b = err.shape[0]
  eye3 = torch.eye(3, device=J.device, dtype=J.dtype).expand(b, 3, 3)
  Jt = J.transpose(1, 2)
  A = torch.bmm(J, Jt) + (float(damping) ** 2) * eye3
  dx = torch.linalg.solve(A, err.unsqueeze(-1)).squeeze(-1)
  dq = torch.bmm(Jt, dx.unsqueeze(-1)).squeeze(-1)
  return torch.clamp(dq, -float(max_dq), float(max_dq))


def dls_toward(
  J: torch.Tensor,
  q: torch.Tensor,
  err: torch.Tensor,
  *,
  damping: float = 0.08,
  max_dq: float = 0.05,
  iterations: int = 3,
  q_min: torch.Tensor | None = None,
  q_max: torch.Tensor | None = None,
) -> torch.Tensor:
  """Multi-iter DLS with linear EE prediction ``ee ← ee + J dq`` inside the loop.

  Caller should pass ``err = p_cmd - p_ee`` once; remaining error is updated
  with the same frozen ``J`` (first-order). For higher fidelity recompute ``J``
  each iter from predicted FK.
  """
  q_out = q
  rem = err
  for _ in range(int(iterations)):
    dq = dls_step(J, rem, damping=damping, max_dq=max_dq)
    q_out = q_out + dq
    rem = rem - torch.bmm(J, dq.unsqueeze(-1)).squeeze(-1)
  if q_min is not None and q_max is not None:
    q_out = torch.maximum(torch.minimum(q_out, q_max), q_min)
  return q_out


def finite_diff_jacobian_check(
  f_ee,
  q: torch.Tensor,
  J: torch.Tensor,
  eps: float = 1e-4,
) -> float:
  """Max relative error between analytic ``J`` and central finite differences.

  ``f_ee(q) -> (B,3)``. Uses env 0 only for a quick sanity check.
  """
  q0 = q[:1].detach().clone()
  n = q0.shape[-1]
  J_fd = torch.zeros(1, 3, n, device=q.device, dtype=q.dtype)
  for i in range(n):
    qp = q0.clone()
    qm = q0.clone()
    qp[0, i] += eps
    qm[0, i] -= eps
    J_fd[0, :, i] = (f_ee(qp) - f_ee(qm))[0] / (2 * eps)
  Ja = J[:1].detach()
  denom = Ja.abs().mean().clamp_min(1e-6)
  return float((J_fd - Ja).abs().mean() / denom)

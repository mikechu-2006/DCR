"""Single-player MinaCalc skill-vector fitting.

The player vector is independent across players: given the frozen global
parameters `theta`, each player's six skill values solve a separate
six-dimensional nonlinear least-squares problem.

The residual is defined in the observed label space

    y_i = log(1 - ACC_i)            # step0/step1 loss
    p_i(x) = log((S_i(x) + T_i(x)) / Max_i)
    r_i(x) = p_i(x) - y_i

where `S+T` is exactly `detail["lost"]` from `core.score` (up to the
torch implementation's iteration count).  The predicted *score* is optionally
clipped to the ACC support used by step1/pairbest, `[0.90, 0.998]`, before
taking `log(1-score)`.  This is needed because `log(1-ACC)` is censored by
the preprocessing filters; without the clip, an easy chart can produce a score
arbitrarily close to 1 and a huge negative loss residual.

The solver is a small Levenberg--Marquardt loop with a finite-difference
Jacobian (six parameters), optional Huber weights, box bounds, and an L2 prior.
It is intentionally single-player: no cross-player state is touched.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# step1/pairbest ACC support: [0.90, 0.998].  The equivalent loss support
# is [log(0.002), log(0.1)] = [-6.2146, -2.3026].
DEFAULT_SCORE_BOUNDS = (0.90, 0.998)
DEFAULT_LOSS_BOUNDS = (math.log(1.0 - DEFAULT_SCORE_BOUNDS[1]),
                       math.log(1.0 - DEFAULT_SCORE_BOUNDS[0]))


def _as_device_tensor(value, *, dtype, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


@torch.no_grad()
def predicted_loss(sc, x, cidx, theta, *, score_bounds=DEFAULT_SCORE_BOUNDS):
    """Return the predicted loss for one player's charts.

    Parameters
    ----------
    sc : TorchScorer
    x : array-like, shape (6,) or (1, 6)
    cidx : array-like, shape (N,)
        Chart indices into `sc` (not beatmap ids).
    theta : Theta
    score_bounds : (lo, hi) or None
        Clip the predicted score to the observed ACC support, e.g.
        `(0.90, 0.998)`.  The returned loss is `log(1 - score)`.

    Returns
    -------
    torch.Tensor, shape (N,)
    """
    device = sc.device
    cidx = _as_device_tensor(cidx, dtype=torch.long, device=device)
    if cidx.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    x = _as_device_tensor(x, dtype=torch.float32, device=device).reshape(1, 6)
    pidx = torch.zeros(cidx.numel(), dtype=torch.long, device=device)
    L = int(sc.n[cidx].max())
    out = sc.forward(x, pidx, cidx, theta, L=L)
    lost = (out["S"] + out["T"]).clamp_min(1e-6)
    score = 1.0 - lost / out["Max"].clamp_min(1e-9)
    if score_bounds is not None:
        score = score.clamp(float(score_bounds[0]), float(score_bounds[1]))
    return torch.log((1.0 - score).clamp_min(1e-12))


@torch.no_grad()
def rmse_of(sc, x, cidx, loss, theta, *, score_bounds=DEFAULT_SCORE_BOUNDS):
    """Return (RMSE, predicted loss) on the supplied observations."""
    pred = predicted_loss(sc, x, cidx, theta, score_bounds=score_bounds)
    y = _as_device_tensor(loss, dtype=torch.float32, device=pred.device)
    r = pred - y
    return float(torch.sqrt((r * r).mean())) if r.numel() else float("nan"), pred


@torch.no_grad()
def fit_player_vector(
    sc,
    cidx,
    loss,
    theta,
    dev=None,
    *,
    x0=None,
    mu=None,
    score_bounds=DEFAULT_SCORE_BOUNDS,
    huber_delta=0.5,
    lam=1e-3,
    iters=12,
    h=1e-3,
    bounds=(0.0, 60.0),
    tol=1e-5,
    verbose=False,
):
    """Fit one player's six-dimensional skill vector.

    Parameters
    ----------
    sc : TorchScorer
    cidx : array-like (N,)
        Chart indices (positions in `sc`'s packed chart tensor).
    loss : array-like (N,)
        Observed loss, `log(1 - ACC)`.
    theta : Theta
        Frozen global parameters (default `Theta()` is MinaCalc 505).
    dev : torch.device or str, optional
        Defaults to `sc.device`.
    x0 : array-like (6,), optional
        Initial vector. Default: all 20.
    mu : array-like (6,), optional
        Center of the L2 prior. Default: all 20.
    score_bounds : (lo, hi) or None
        Clip the predicted score to the observed ACC support, e.g.
        `(0.90, 0.998)`.  Use `None` for an unclipped fit.
    huber_delta : float or None
        Huber threshold in loss units. `None` uses plain squared error.
    lam : float
        L2 prior strength (sum over dimensions).
    iters : int
        Maximum LM iterations.
    h : float
        Finite-difference step.
    bounds : (lo, hi)
        Skill bounds.
    tol : float
        Stop when the accepted step's max component is below this.
    verbose : bool
        Print one line per iteration.

    Returns
    -------
    (x, info)
        `x` is a (6,) tensor on `dev`; `info` contains `rmse`, `n`,
        `cost`, `iters`, and the final `active` mask.
    """
    device = sc.device if dev is None else torch.device(dev)
    cidx = _as_device_tensor(cidx, dtype=torch.long, device=device)
    y = _as_device_tensor(loss, dtype=torch.float32, device=device)
    if cidx.numel() == 0:
        raise ValueError("fit_player_vector needs at least one observation")

    x = (torch.full((6,), 20.0, device=device, dtype=torch.float32)
         if x0 is None else _as_device_tensor(x0, dtype=torch.float32,
                                              device=device).reshape(6).clone())
    mu = (torch.full((6,), 20.0, device=device, dtype=torch.float32)
          if mu is None else _as_device_tensor(mu, dtype=torch.float32,
                                               device=device).reshape(6).clone())
    lo, hi = float(bounds[0]), float(bounds[1])
    x.clamp_(lo, hi)
    L = int(sc.n[cidx].max())
    pidx = torch.zeros(cidx.numel(), dtype=torch.long, device=device)

    def predict_rows(X, cidx_b, pidx_b):
        out = sc.forward(X, pidx_b, cidx_b, theta, L=L)
        lost = (out["S"] + out["T"]).clamp_min(1e-6)
        score = 1.0 - lost / out["Max"].clamp_min(1e-9)
        if score_bounds is not None:
            score = score.clamp(float(score_bounds[0]), float(score_bounds[1]))
        return torch.log((1.0 - score).clamp_min(1e-12))

    def predict(xv):
        return predict_rows(xv.reshape(1, 6), cidx, pidx)

    def rho_weights(r):
        if huber_delta is None:
            return 0.5 * r * r, torch.ones_like(r)
        a = r.abs()
        d = float(huber_delta)
        rho = torch.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))
        w = torch.where(a <= d, torch.ones_like(a), d / a.clamp_min(1e-12))
        return rho, w

    def objective(xv):
        r = predict(xv) - y
        rho, _ = rho_weights(r)
        return float(rho.sum()) + lam * float(((xv - mu) ** 2).sum()), r

    cost, r = objective(x)
    best_x, best_cost = x.clone(), cost
    tau = 1e-3
    J = torch.zeros((cidx.numel(), 6), device=device, dtype=torch.float32)
    eye = torch.eye(6, device=device, dtype=torch.float32)
    n_acc = 0
    n_iter = 0
    for it in range(int(iters)):
        n_iter = it + 1
        # Evaluate the base point and all six finite-difference perturbations
        # in one batched forward.  This is ~7x fewer Python/torch calls than
        # six independent forwards, while doing the same arithmetic.
        X7 = torch.empty((7, 6), device=device, dtype=torch.float32)
        X7[0] = x
        X7[1:] = x + h * eye
        cidx7 = cidx.repeat(7)
        pidx7 = torch.arange(7, device=device).repeat_interleave(cidx.numel())
        preds = predict_rows(X7, cidx7, pidx7).reshape(7, cidx.numel())
        r = preds[0] - y
        J = (preds[1:] - preds[0]).T / h
        rho, w = rho_weights(r)
        cost = float(rho.sum()) + lam * float(((x - mu) ** 2).sum())
        JtWJ = J.T @ (w[:, None] * J)
        g = J.T @ (w * r)
        A = (JtWJ + lam * torch.eye(6, device=device)
             + torch.diag(torch.full((6,), tau, device=device)))
        b = -(g + lam * (x - mu))
        try:
            dx = torch.linalg.solve(A, b)
        except Exception:
            break
        x_new = torch.clamp(x + dx, lo, hi)
        cost_new, r_new = objective(x_new)
        if cost_new < cost:
            x, r, cost = x_new, r_new, cost_new
            n_acc += 1
            if cost < best_cost:
                best_x, best_cost = x.clone(), cost
            tau = max(tau / 3.0, 1e-6)
            if float(dx.abs().max()) < tol:
                if verbose:
                    print(f"  lm it={it + 1:02d} cost={cost:.6f} "
                          f"x={np.round(x.detach().cpu().numpy(), 3).tolist()}",
                          flush=True)
                break
        else:
            tau = min(tau * 10.0, 1e8)
        if verbose:
            print(f"  lm it={it + 1:02d} cost={cost:.6f} tau={tau:.2e} "
                  f"x={np.round(x.detach().cpu().numpy(), 3).tolist()}", flush=True)

    # Use the best accepted point, not a rejected trial.
    if best_cost < cost:
        x = best_x
    rmse = float(torch.sqrt(((predict(x) - y) ** 2).mean()))
    active = (J.norm(dim=0) > 1e-4 * math.sqrt(cidx.numel())).cpu().numpy()
    info = {
        "rmse": rmse,
        "n": int(cidx.numel()),
        "cost": float(best_cost),
        "accepted_steps": int(n_acc),
        "iters": int(n_iter),
        "active": active.tolist(),
        "score_bounds": None if score_bounds is None else [float(score_bounds[0]), float(score_bounds[1])],
    }
    return x, info

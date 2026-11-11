# Stage-3 BPTT Reward Framework (Homie G1 / mjlab)

Self-contained note for external analysis (ChatGPT etc.).  
Code paths: `HomieRL_mjlab/src/homierl_mjlab/rl/stage3_bptt/` + Phase-3 PPO rewards in `mdp/rewards.py` / `env_cfgs.py`.

---

## 1. System overview (what is being optimized)

Two **separate** reward systems run in parallel:

| System | Used for | Optimizes? |
|--------|----------|------------|
| **A. Stage3 BPTT reward** `stage3_reward(...)` | Replay + imagination | **Yes** — reach/SDF/stability (reach off in body phase) |
| **B. Phase-1/2 loco PPO step reward** | Blended into replay as `w_loco * ppo_rew` | **Yes** — same env Episode_Reward stack (tracking, height, …) |
| **C. Teacher MSE** | Actor loss regularizer | **Yes** — not the only loco signal; body phase also **executes teacher** in sim |

Training loop (each iter):

1. **Collect**: body phase → teacher drives loco (`body_exec_teacher`); arm phase → actor body + palm-delta. Residual EE = 0.
2. Store `reward = stage3_reward + w_loco * ppo_rew` (+ teacher_body for BC).
3. Fit **world model** on real `(s,u)→s'`.
4. Fit **critic** with GAE on blended reward (loco + stage3).
5. **Actor BPTT**: imag `J` (stage3 terms) + `w_teacher * MSE(a_body, teacher)`.

Control mode (default): `body_ee_palm_delta` (`action_semantics=body14_ee3_palm_delta_v1`)

- `a_ee`: tanh × `ee_delta_scale_m` (0.03 m) in Homie **base**; `p_sub_B = palm_B + delta_B`; rate-limit in world.
- `a_body`: applied to loco joints (`q = q_default + 0.25 * a`); supervised by frozen phase-2 HIM (cannot copy HIM weights into Stage3 MLP — obs mismatch).
- True task goal `p_goal` is **not** overwritten for reward; only `p_sub` is injected into grasp for IK/loco obs.
- Legacy `hierarchical_subgoal` / `ee_residual` remain for ablation; **do not mix** old 3-D replay / ckpts.

Imagination gap (important for analysis):

- After WM step, palm for reward uses a **first-order kinematic `palm_step_fn`** = current palm + delta (approximate vs full arm FK).
- Imagination gates safe/margin with stored `present` when `imag_use_present=True`.
- Soft survival from predicted tilt when `imag_soft_survival=True` (not a hard truncate).

---

## 2. Stage3 reward (the one that trains the BPTT actor)

Source: `rl/stage3_bptt/rewards_stage3.py`  
Defaults / weights: `rl/stage3_bptt/cfg.py`

### 2.1 Scalar formula

Per env-step (batch of envs):

```text
r = w_target * R_target
  + w_safe   * R_safe     # gated by present in real collect
  + w_margin * R_margin   # gated by present in real collect
  + w_smooth * R_smooth
  + w_stability * R_stab  # currently w_stability=0, unused
```

Current weights:

| Weight | Value | Notes |
|--------|------:|-------|
| `w_target` | 1.0 | Reach attractiveness |
| `w_safe` | 1.5 | Collision / under-clearance penalty |
| `w_margin` | 0.5 | Dense bonus for staying above `d_safe` |
| `w_smooth` | 0.05 | Action smoothness |
| `w_stability` | 0.0 | Off |

Geometry params:

| Param | Value | Meaning |
|-------|------:|---------|
| `sigma_p` | 0.08 m | Soft reach kernel length |
| `d_safe` | 0.05 m | Desired clearance (meters) |
| `d_max` | 0.40 m | Clearance normalize scale |
| `reward_clip` | ±5.0 | Applied only in **imagination** BPTT |

### 2.2 Terms (math)

**Target (reach):**

```text
R_target = exp( - ‖palm − goal‖₂ / σ_p )
```

Range ≈ `(0, 1]`. Dense; no sparse success bonus in Stage3 reward.

**Safe (penetration / under-margin penalty):**

Clearance input `c` is normalized: `c = (φ − r_sphere) / d_max` (same convention as `arm_clearance_field`).

```text
c_m = c * d_max                    # meters
viol_n = clamp(d_safe − c_m, 0) / d_safe
R_safe = − max_j (viol_n_j²)       # worst sphere, not mean
```

- Soft hinge under `d_safe`; quadratic in normalized violation.
- Worst-of-spheres so one link clipping the rim is not washed out.

**Margin (dense “stay clear” shaping):**

```text
worst = min_j c_m
R_margin = clamp( (worst − d_safe) / d_safe , 0, 1 )
```

- 0 at/under `d_safe`, 1 at ≥ `2·d_safe`.

**Smooth:**

```text
Δ = clamp(a − a_prev, −0.2, 0.2)
R_smooth = − ‖Δ‖₂²
```

Clamp prevents hierarchical large offsets from exploding this term.

**Present gating (real collect only):**

```text
if container.present:
  R_safe   *= 1
  R_margin *= 1
else:
  R_safe = R_margin = 0
```

No crate → no clearance terms; reach + smooth still active.

### 2.3 Clearance sources (real vs imagination)

| Context | Clearance used |
|---------|----------------|
| Real collect | Active-arm 3 spheres from `arm_clearance_field` (L or R) |
| Imagination | `palm_probe_clearance`: palm + up/back probes vs analytic open-box SDF from packed `geom` (10-D container cmd) |

Imagination is **not** full 6-sphere FK; only palm probes.

### 2.4 Where `stage3_reward` is consumed

1. **Collector** → `Train/mean_reward` (TB), replay `reward`.
2. **Critic TD / GAE** on replay rewards (clipped ±`reward_clip` in critic update).
3. **Actor imagination** → `Train/imagination_J` = mean of `J` (maximize).

Actor objective:

```text
J = Σ_{t=0}^{H−1} γ^t · r_t + γ^H · V(s_H)
loss_actor = − mean(J)
```

`H = imagine_horizon = 8`, `γ = 0.99`.

---

## 3. Phase-3 PPO rewards (monitor only during BPTT)

These **do not** train the Stage3 actor. They come from the underlying Phase-3 env reward manager while frozen loco runs. Logged as `Episode_Reward/*` / TB `Episode/*`.

Phase ≥ 3 additions (`env_cfgs.py`):

| Term | Weight | Role |
|------|-------:|------|
| `reach_ee_position` | 2.0 | Soft reach `exp(−‖err‖²/σ²)`, σ=0.08 (**differs** from Stage3 `exp(−d/σ)`) |
| `clearance_penalty` | 2.5 | Crate clearance / penetration (d_safe=0.05, d_max=0.40) |
| `reach_ee_progress` | 1.0 | Progress toward grasp |
| `reach_success` | 0.5 | Sparse-ish success (threshold 0.03 m) |
| `ee_residual_l2` | 0.05 | In hierarchical mode residual≈0 → ~0 |

Plus full Homie loco terms (tracking vel/height, orientation, feet, torques, `no_fly`, `stand_sway`, …).  
`Train/ppo_step_reward` = mean **step** PPO reward during collect (loco health proxy).

---

## 4. Terminations (env, not Stage3-shaped)

| Name | Type | Condition |
|------|------|-----------|
| `time_out` | truncation | `episode_length_s = 20` → ~1000 policy steps @ 50 Hz |
| `fell_over` | termination | `bad_orientation`, limit **70°** |

No terminate-on-collision. Collision only via clearance penalties.

Curriculum B: container `presence_prob ≈ 0.35`, often relocate grasp under open-top for lift-over.

---

## 5. Observed TB behavior (~1.5k–1.7k iters, for context)

Typical recent run symptoms (not claims of correctness):

- `Train/mean_reward` (Stage3): early rise then plateau / slight drop (~−0.07…−0.09).
- `Train/neg_clearance_frac`: early drop then flat ~0.03–0.045.
- `Train/imagination_J`: trending **down** (imagined return worsens).
- `Train/ppo_step_reward`: slow decline (frozen loco stressed by changing subgoals?).
- `Train/nan_streak = 0`.
- Episode-length spikes every ~30 iters: 4096 envs syncing on `time_out` + collect 32 steps/iter (logging artifact).

---

## 6. Design tensions / questions for analysis

1. **Dual reward worlds**: BPTT optimizes `stage3_reward`; dashboards also show PPO `clearance_penalty` / `reach_*` with different weights (2.5 / 2.0 vs Stage3 1.5 / 1.0). Easy to over-interpret PPO curves as “policy learning.”
2. **Scale balance**: `R_target∈(0,1]`, `R_safe∈[−1,0]`, `R_margin∈[0,1]`. With `w_safe=1.5`, frequent mild penetration can dominate mean reward negativity.
3. **Present ≈ 0.28**: most steps have **no** crate → Stage3 safe/margin often zero; sparse collision signal for BPTT.
4. **Imagination mismatch**: kinematic palm shortcut + always-on safe/margin + no termination may make `J` poorly correlated with real `mean_reward`.
5. **No Stage3 success bonus**: only soft `exp(-d/σ)`; PPO has `reach_success` but actor never sees it.
6. **Smooth on hierarchical offsets**: actions are absolute-ish offsets up to 0.55 m; even with Δ clamp 0.2, smoothness may fight large corrective subgoals.
7. **Frozen loco**: bad subgoals hurt PPO tracking / fall rate without Stage3 reward explicitly rewarding stability (`w_stability=0`).

---

## 7. Minimal code reference (Stage3 only)

```python
# rewards_stage3.py (conceptual)
R_target = exp(-||palm - goal|| / sigma_p)
R_safe   = -max_j( (clamp(d_safe - c_m, 0)/d_safe)^2 )
R_margin = clamp( (min_j(c_m) - d_safe) / d_safe , 0, 1 )
R_smooth = -||clamp(a - a_prev, -0.2, 0.2)||^2

r = 1.0*R_target + 1.5*R_safe + 0.5*R_margin + 0.05*R_smooth
# real collect: R_safe, R_margin *= present
# imagination: clamp(r, -5, 5); present gating omitted
```

Files:

- `rl/stage3_bptt/rewards_stage3.py` — Stage3 terms
- `rl/stage3_bptt/cfg.py` — weights / geometry
- `rl/stage3_bptt/collector.py` — real `stage3_reward` + present gate
- `rl/stage3_bptt/trainer.py` — imagination `stage3_reward` + BPTT `J`
- `rl/stage3_bptt/geom_clearance.py` — imagination palm probes
- `mdp/rewards.py` — PPO `reach_*` / `clearance_penalty` (monitor)
- `env_cfgs.py` — Phase-3 PPO weights + terminations

---

## 8. Ask ChatGPT (suggested prompts)

- Is the weight balance (`w_safe` vs `w_target` vs `w_margin`) likely to produce plateaued `neg_clearance` and flat/negative `mean_reward`?
- Given present≈0.28, how should clearance terms be reweighted or resampled?
- Does maximizing imagination `J` with kinematic palm + ungated safe terms explain declining `imagination_J` while real reward plateaus?
- What sparse / progress / stability terms would you add without destabilizing frozen loco?
- How to align Stage3 reward with PPO `clearance_penalty` / `reach_success` for apples-to-apples monitoring?

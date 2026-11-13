# Stage-3 BPTT interface table (Homie G1 Inspire-waist)

Verified against `HomieRL_mjlab` source (not copied from other robots).

## Control / timing

| Item | Value | Source |
|------|-------|--------|
| Policy dt | `sim.dt * decimation` ≈ 0.02 s (50 Hz) | `env_cfgs` `decimation=4` |
| Phase-2 actions | 14: 12 legs + `waist_roll` + `waist_pitch` | `LOCO_JOINT_NAMES` |
| Phase-2 held | `waist_yaw` PD at default | `hold_waist_yaw_default` |
| Action semantics | `q_target = q_default + 0.25 * a` (tanh/Gaussian) | `HOMIE_ACTION_SCALE` |
| Phase-3 PPO EE | 3-D residual → DLS → arm joint targets | `EeResidualIkAction` |
| Actor history (HIM) | 6 steps one-step obs | `him.actor_history_length` |

## Phase-2 one-step actor obs (approx.)

`proprio(68) | last_action(14) | grasp(7) | container(10)` — exact packing in `env_cfgs` / `observations.py`.  
Phase-3 PPO packs `a17 | grasp7 | sdf | clearance | ee_err` → 229-D one-step.

## Stage-3 BPTT actor (this package)

| Field | Dim | Unit / frame | Notes |
|-------|-----|--------------|-------|
| `control_mode` | — | cfg | **`body_ee_palm_delta` (default)**; legacy: `hierarchical_subgoal`, `ee_residual` |
| progressive curriculum | — | auto (default) | **body** (arm locked, teacher exec, no box) → **arm** (reach≥0.5) → **box** (presence +0.05 when clear-reach≥0.3, max 0.35) |
| loco rewards in Stage3 | — | `w_loco * ppo_rew` | Phase-1/2 env PPO step reward **blended into replay** (not monitor-only); teacher MSE is regularizer |
| `action_semantics` | — | ckpt gate | `body14_ee3_palm_delta_v4_loco_hist6` — refuse silent load of old 3-D / v3 1-frame ckpts |
| body+EE action | 17 | `a_body(14)` raw ±100 + `a_ee(3)` m **base** | `p_sub_B = palm_B + delta_B`; rate-limit in world; inject `p_sub`; EE residual = 0 |
| body decode | 14 | loco joints | `q = q_default + 0.25 * a_body` on `LOCO_JOINT_NAMES`; arms via IK only |
| teacher | 14 | frozen phase-2/3 HIM | MSE(`a_body`, teacher) on **same 6-frame loco hist** as student `body_net`; `w_teacher` |
| Stage3 loco input | `6 × one_step` | HIM stack | `body_net` sees flat `history_length × one_step` (default 6×229), aligned with teacher `act_inference` |
| Stage3 obs (body_ee) | `loco_hist ‖ reach` | base | reach = palm, goal, goal−palm, prev_sub−palm, last_a(17), SDF, clearance |
| hierarchical action (legacy) | 3 | meters, **world** offset from episode `p_ee0` | `p_sub = rate_limit(p_ee0 + a)`; EE residual = 0 |
| legacy `delta_p` | 3 | meters, **base** | `p_cmd = p_palm + scale * tanh(raw)` on EE IK only |
| Stage3 obs (hierarchical) | `15+K+6` | base | palm, true goal, goal−ee0, palm−ee0, last offset, SDF, clearance |
| Applied control logged | full joint position targets | rad | same PD interface as sim |
| Arm DoF | 7 per arm (`shoulder×3, elbow, wrist×3`) | | v1: single arm (configurable left/right); wrists often soft-locked |

## State vector (world-model / critic privilege)

Packed by `state_adapter.pack_robot_state`:

`q(n_j) | qd(n_j) | root_pos(3) | root_quat_wxyz(4) | root_lin_vel_b(3) | root_ang_vel_b(3) | palm_pos_b(3)`

`n_j` from the loaded entity (Inspire-waist ≈ 29 actuated DOF in MJCF; confirm at runtime).

## Geometry

| Item | v1 default | Notes |
|------|------------|-------|
| SDF queries | K=64 or 256 (cfg) | base-frame grid → world via pelvis yaw |
| Clearance spheres | 6 (L/R shoulder, elbow, palm) | `sdf_observations._ARM_SPHERES` |
| Container | analytic open square walls | `sdf_geometry` |
| d_max / d_safe | 0.40 / 0.05 m | cfg |

## Gradient / freeze rules (BPTT)

- Stage-2 loco actor: `requires_grad_(False)` weights; **no** `torch.no_grad()` on forward if obs depend on predicted state (v1 uses open-loop logged loco cmds — see LIMITATIONS).
- World model / target critic: frozen during actor BPTT; input grads kept.
- Critic TD targets: detached.
- Imagination must recompute FK/J/SDF from **predicted** state (not stale sim buffers).

## LIMITATIONS (v1)

1. Imagination uses **logged stage-2 joint commands** (open-loop) rather than full HIM history rebuild through the frozen PPO — documented stop-gap; real rollouts still run frozen stage-2 online.
2. Differentiable arm FK/J uses geometric axes + measured link origins snapshotted / predicted palm from WM; full MuJoCo autodiff FK is not required for first BPTT loop.
3. Contact recovery is **not** modeled; hard collision → terminate in real env; imagination uses SDF risk on **palm probes** from packed geom (not full 6-sphere FK).
4. Curriculum **B** enables crates (`presence_prob` cfg); **A** is reach-only. Safe reward uses worst-sphere breach (aligned with PPO `clearance_penalty`).
5. Existing `Mjlab-Homie-Flat-Unitree-G1-Phase3` PPO path is unchanged.

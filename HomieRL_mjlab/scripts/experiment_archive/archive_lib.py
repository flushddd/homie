"""Training-run archive helpers.

Reads only the resolved configs that mjlab already dumped under
``<run>/params/{env,agent}.yaml``. Never invents historical values from
current source code.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from yaml.loader import SafeLoader

UNRECORDED = "未记录"


def _unique_archive_dir(archives_root: Path, archive_id: str) -> Path:
  """Return a non-existing archive path; suffix ``_2``, ``_3``, … on collision."""
  candidate = archives_root / archive_id
  if not candidate.exists():
    return candidate
  n = 2
  while True:
    alt = archives_root / f"{archive_id}_{n}"
    if not alt.exists():
      return alt
    n += 1

# Fields we try to surface with provenance for report / compare.
KEY_PARAM_PATHS = (
  ("seed_env", ("env", "seed")),
  ("seed_agent", ("agent", "seed")),
  ("decimation", ("env", "decimation")),
  ("sim_timestep", ("env", "sim", "mujoco", "timestep")),
  ("action_scale", ("env", "actions", "joint_pos", "scale")),
  ("actuator_names_policy", ("env", "actions", "joint_pos", "actuator_names")),
  ("num_envs", ("env", "scene", "num_envs")),
  ("episode_length_s", ("env", "episode_length_s")),
  ("experiment_name", ("agent", "experiment_name")),
  ("run_name", ("agent", "run_name")),
  ("max_iterations", ("agent", "max_iterations")),
  ("num_steps_per_env", ("agent", "num_steps_per_env")),
  ("save_interval", ("agent", "save_interval")),
  ("resume", ("agent", "resume")),
  ("load_run", ("agent", "load_run")),
  ("load_checkpoint", ("agent", "load_checkpoint")),
  ("algorithm", ("agent", "algorithm")),
  ("him", ("agent", "him")),
  ("learning_rate", ("agent", "algorithm", "learning_rate")),
  ("clip_param", ("agent", "algorithm", "clip_param")),
  ("gamma", ("agent", "algorithm", "gamma")),
  ("lam", ("agent", "algorithm", "lam")),
  ("entropy_coef", ("agent", "algorithm", "entropy_coef")),
  ("desired_kl", ("agent", "algorithm", "desired_kl")),
)


class _TagSafeLoader(SafeLoader):
  """Load mjlab dumps that contain !!python/* tags without importing mjlab."""


def _construct_any(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
  if isinstance(node, yaml.ScalarNode):
    return loader.construct_scalar(node)
  if isinstance(node, yaml.SequenceNode):
    return loader.construct_sequence(node)
  if isinstance(node, yaml.MappingNode):
    return loader.construct_mapping(node)
  return None


def _construct_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple:
  return tuple(loader.construct_sequence(node))


_TagSafeLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/name", _construct_any
)
_TagSafeLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/object/apply", _construct_any
)
_TagSafeLoader.add_multi_constructor(
  "tag:yaml.org,2002:python/object", _construct_any
)
_TagSafeLoader.add_constructor(
  "tag:yaml.org,2002:python/tuple", _construct_tuple
)


def load_mjlab_yaml(path: Path) -> Any:
  text = path.read_text(encoding="utf-8")
  return yaml.load(text, Loader=_TagSafeLoader)


def dig(data: Any, *keys: str, default: Any = UNRECORDED) -> Any:
  cur = data
  for k in keys:
    if not isinstance(cur, dict) or k not in cur:
      return default
    cur = cur[k]
  if cur is None:
    return UNRECORDED
  return cur


def jsonable(obj: Any) -> Any:
  if obj is UNRECORDED:
    return UNRECORDED
  if isinstance(obj, Path):
    return str(obj)
  if isinstance(obj, dict):
    return {str(k): jsonable(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [jsonable(v) for v in obj]
  if isinstance(obj, (str, int, float, bool)) or obj is None:
    return obj
  return str(obj)


@dataclass(frozen=True)
class RunPaths:
  run_dir: Path
  env_yaml: Path
  agent_yaml: Path

  @classmethod
  def from_run_dir(cls, run_dir: Path) -> "RunPaths":
    run_dir = run_dir.resolve()
    env_yaml = run_dir / "params" / "env.yaml"
    agent_yaml = run_dir / "params" / "agent.yaml"
    if not env_yaml.is_file() or not agent_yaml.is_file():
      raise FileNotFoundError(
        f"Missing mjlab params dump under {run_dir}/params "
        f"(need env.yaml + agent.yaml). Refusing to invent values."
      )
    return cls(run_dir=run_dir, env_yaml=env_yaml, agent_yaml=agent_yaml)


def file_sha256(path: Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def git_snapshot(repo_root: Path) -> dict[str, Any]:
  """Capture git state *now* (archive time). Train-time git is usually absent."""

  def _run(args: list[str]) -> tuple[int, str]:
    try:
      p = subprocess.run(
        args,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
      )
      return p.returncode, (p.stdout or "").strip()
    except FileNotFoundError:
      return 127, ""

  out: dict[str, Any] = {
    "repo_root": str(repo_root),
    "captured_at": datetime.now(timezone.utc).isoformat(),
    "note": (
      "This is the git state at archive time, not necessarily the exact "
      "commit used when the run started. Train-time git dump was not found "
      "in the run directory."
    ),
  }
  code, head = _run(["git", "rev-parse", "HEAD"])
  out["head"] = head if code == 0 and head else UNRECORDED
  code, branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
  out["branch"] = branch if code == 0 and branch else UNRECORDED
  code, status = _run(["git", "status", "--porcelain"])
  if code == 0:
    out["dirty"] = bool(status)
    out["status_porcelain"] = status.splitlines() if status else []
  else:
    out["dirty"] = UNRECORDED
    out["status_porcelain"] = UNRECORDED
  code, diff_stat = _run(["git", "diff", "--stat", "HEAD"])
  out["diff_stat"] = diff_stat if code == 0 else UNRECORDED
  return out


def extract_actuators(env: dict) -> list[dict[str, Any]]:
  arts = dig(
    env, "scene", "entities", "robot", "articulation", "actuators", default=[]
  )
  if arts is UNRECORDED or not isinstance(arts, (list, tuple)):
    return []
  rows: list[dict[str, Any]] = []
  for i, act in enumerate(arts):
    if not isinstance(act, dict):
      rows.append(
        {
          "index": i,
          "target_names_expr": UNRECORDED,
          "stiffness_Kp": UNRECORDED,
          "damping_Kd": UNRECORDED,
          "source": "params/env.yaml:scene.entities.robot.articulation.actuators",
        }
      )
      continue
    rows.append(
      {
        "index": i,
        "target_names_expr": act.get("target_names_expr", UNRECORDED),
        "stiffness_Kp": act.get("stiffness", UNRECORDED),
        "damping_Kd": act.get("damping", UNRECORDED),
        "effort_limit": act.get("effort_limit", UNRECORDED),
        "source": (
          "params/env.yaml:scene.entities.robot.articulation.actuators"
          f"[{i}]"
        ),
      }
    )
  return rows


def extract_rewards(env: dict) -> dict[str, Any]:
  rewards = dig(env, "rewards", default=UNRECORDED)
  if rewards is UNRECORDED or not isinstance(rewards, dict):
    return {"items": UNRECORDED, "source": "params/env.yaml:rewards"}
  items = {}
  for name, cfg in rewards.items():
    if not isinstance(cfg, dict):
      items[name] = {
        "weight": UNRECORDED,
        "params": UNRECORDED,
        "source": f"params/env.yaml:rewards.{name}",
      }
      continue
    items[name] = {
      "weight": cfg.get("weight", UNRECORDED),
      "params": cfg.get("params", UNRECORDED),
      # func is stripped by tag-safe loader → often empty string
      "func": cfg.get("func", UNRECORDED) or UNRECORDED,
      "source": f"params/env.yaml:rewards.{name}",
    }
  return {"items": items, "source": "params/env.yaml:rewards"}


def extract_key_params(env: dict, agent: dict) -> dict[str, Any]:
  root = {"env": env, "agent": agent}
  out: dict[str, Any] = {}
  for name, path in KEY_PARAM_PATHS:
    val = dig(root, *path)
    out[name] = {
      "value": jsonable(val),
      "source": "params/"
      + ("env.yaml" if path[0] == "env" else "agent.yaml")
      + ":"
      + ".".join(path[1:]),
    }

  # Default joint angles + names from init_state.
  joint_pos = dig(
    env, "scene", "entities", "robot", "init_state", "joint_pos", default=UNRECORDED
  )
  out["default_joint_angles"] = {
    "value": jsonable(joint_pos),
    "source": "params/env.yaml:scene.entities.robot.init_state.joint_pos",
  }
  if isinstance(joint_pos, dict):
    names = [k for k in joint_pos.keys() if k != ".*"]
    out["joint_names_from_init_state"] = {
      "value": names,
      "source": "params/env.yaml:scene.entities.robot.init_state.joint_pos (keys)",
    }
  else:
    out["joint_names_from_init_state"] = {
      "value": UNRECORDED,
      "source": "params/env.yaml:scene.entities.robot.init_state.joint_pos (keys)",
    }

  out["actuators_Kp_Kd"] = {
    "value": jsonable(extract_actuators(env)),
    "source": "params/env.yaml:scene.entities.robot.articulation.actuators",
  }
  out["rewards"] = {
    "value": jsonable(extract_rewards(env)),
    "source": "params/env.yaml:rewards",
  }

  # Convenience: control dt if both present.
  dt = dig(env, "sim", "mujoco", "timestep", default=UNRECORDED)
  dec = dig(env, "decimation", default=UNRECORDED)
  if isinstance(dt, (int, float)) and isinstance(dec, (int, float)):
    out["control_dt"] = {
      "value": float(dt) * float(dec),
      "source": "derived: env.sim.mujoco.timestep * env.decimation",
    }
  else:
    out["control_dt"] = {
      "value": UNRECORDED,
      "source": "derived: env.sim.mujoco.timestep * env.decimation",
    }
  return out


def list_checkpoints(run_dir: Path) -> list[str]:
  return sorted(p.name for p in run_dir.glob("model_*.pt"))


def infer_task_hint(agent: dict, run_dir: Path) -> str:
  run_name = agent.get("run_name") or ""
  name = run_dir.name
  if "phase2" in str(run_name) or "phase2" in name:
    return "Mjlab-Homie-Flat-Unitree-G1-Phase2 (inferred from run_name/dir only)"
  if "phase1" in str(run_name) or name == "phase1":
    return "Mjlab-Homie-Flat-Unitree-G1 (inferred from run_name/dir only)"
  return UNRECORDED


def build_manifest(
  *,
  run: RunPaths,
  env: dict,
  agent: dict,
  repo_root: Path,
  archive_dir: Path,
  key_params: dict[str, Any],
) -> dict[str, Any]:
  return {
    "schema_version": 1,
    "archive_kind": "train",
    "archived_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_dir": str(run.run_dir),
    "archive_dir": str(archive_dir),
    "task": infer_task_hint(agent, run.run_dir),
    "robot": {
      "value": "Unitree G1 inspire-waist (from env scene entity)",
      "entity": "robot",
      "spec_hint": dig(
        env, "scene", "entities", "robot", "spec_fn", default=UNRECORDED
      ),
      "source": "params/env.yaml:scene.entities.robot",
    },
    "seeds": {
      "env": dig(env, "seed"),
      "agent": dig(agent, "seed"),
      "source_env": "params/env.yaml:seed",
      "source_agent": "params/agent.yaml:seed",
    },
    "code_version": git_snapshot(repo_root),
    "train_time_git_in_run_dir": UNRECORDED,
    "params_snapshot": {
      "env_yaml": {
        "path": str(run.env_yaml),
        "sha256": file_sha256(run.env_yaml),
        "source": "mjlab.scripts.train dump_yaml(log_dir/params/env.yaml)",
      },
      "agent_yaml": {
        "path": str(run.agent_yaml),
        "sha256": file_sha256(run.agent_yaml),
        "source": "mjlab.scripts.train dump_yaml(log_dir/params/agent.yaml)",
      },
    },
    "checkpoints_in_run_dir": list_checkpoints(run.run_dir),
    "cli_overrides": {
      "value": UNRECORDED,
      "note": (
        "mjlab does not persist argv separately; CLI overrides are already "
        "baked into params/{env,agent}.yaml at train start."
      ),
      "source": "params/{env,agent}.yaml (resolved)",
    },
    "key_params": key_params,
    "evidence_classes_note": (
      "Reports must separate 数据观察 / 原因推测 / 已验证结论."
    ),
  }


def flatten_for_diff(key_params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
  flat: dict[str, Any] = {}
  for k, meta in key_params.items():
    path = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
    if not isinstance(meta, dict) or "value" not in meta:
      flat[path] = jsonable(meta)
      continue
    val = meta["value"]
    if k == "rewards" and isinstance(val, dict) and isinstance(val.get("items"), dict):
      for rk, rv in val["items"].items():
        if isinstance(rv, dict):
          flat[f"{path}.items.{rk}.weight"] = rv.get("weight", UNRECORDED)
          flat[f"{path}.items.{rk}.params"] = jsonable(rv.get("params", UNRECORDED))
        else:
          flat[f"{path}.items.{rk}"] = jsonable(rv)
    elif k == "actuators_Kp_Kd" and isinstance(val, list):
      for row in val:
        if not isinstance(row, dict):
          continue
        idx = row.get("index", "?")
        flat[f"{path}[{idx}].target_names_expr"] = jsonable(
          row.get("target_names_expr", UNRECORDED)
        )
        flat[f"{path}[{idx}].stiffness_Kp"] = row.get("stiffness_Kp", UNRECORDED)
        flat[f"{path}[{idx}].damping_Kd"] = row.get("damping_Kd", UNRECORDED)
    elif isinstance(val, dict) and k in {"algorithm", "him"}:
      for sk, sv in val.items():
        flat[f"{path}.{sk}"] = jsonable(sv)
    else:
      flat[path] = jsonable(val)
  return flat


def _as_key_params_dict(value: Any) -> dict[str, Any] | None:
  """Return key_params mapping, or None if missing / 未记录 / non-dict."""
  if value is None or value == UNRECORDED:
    return None
  if isinstance(value, dict):
    return value
  return None


def diff_key_params(
  current: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, Any]:
  previous = _as_key_params_dict(previous)
  if previous is None:
    return {
      "previous_archive": None,
      "changed": [],
      "only_current": [],
      "only_previous": [],
      "note": "No previous archive found for comparison.",
    }
  if not isinstance(current, dict):
    current = {}
  a = flatten_for_diff(current)
  b = flatten_for_diff(previous)
  keys = sorted(set(a) | set(b))
  changed = []
  only_c = []
  only_p = []
  for k in keys:
    if k not in b:
      only_c.append({"key": k, "current": a[k]})
    elif k not in a:
      only_p.append({"key": k, "previous": b[k]})
    elif a[k] != b[k]:
      changed.append({"key": k, "previous": b[k], "current": a[k]})
  return {
    "changed": changed,
    "only_current": only_c,
    "only_previous": only_p,
  }


def render_report(
  manifest: dict[str, Any],
  diff: dict[str, Any],
  previous_id: str | None,
) -> str:
  kp = manifest["key_params"]
  lines: list[str] = []
  lines.append(f"# Train archive report: `{Path(manifest['run_dir']).name}`")
  lines.append("")
  lines.append(f"- Archived at (UTC): `{manifest['archived_at_utc']}`")
  lines.append(f"- Run dir: `{manifest['run_dir']}`")
  lines.append(f"- Archive dir: `{manifest['archive_dir']}`")
  lines.append("")
  lines.append("## 数据观察")
  lines.append("")
  lines.append(
    "下列内容均来自该次训练目录下 mjlab 已落盘的 "
    "`params/env.yaml` / `params/agent.yaml`，或归档时刻的 git 快照。"
  )
  lines.append("")
  lines.append(f"- Task hint: `{manifest['task']}`")
  lines.append(
    f"- Seeds: env=`{manifest['seeds']['env']}`, agent=`{manifest['seeds']['agent']}`"
  )
  lines.append(
    f"- Git HEAD (archive-time): `{manifest['code_version'].get('head', UNRECORDED)}`"
  )
  lines.append(
    f"- Git dirty (archive-time): `{manifest['code_version'].get('dirty', UNRECORDED)}`"
  )
  lines.append(
    f"- Train-time git in run dir: `{manifest['train_time_git_in_run_dir']}`"
  )
  lines.append(
    f"- Checkpoints present: `{', '.join(manifest['checkpoints_in_run_dir']) or UNRECORDED}`"
  )
  lines.append(
    f"- decimation: `{kp['decimation']['value']}` "
    f"(source: `{kp['decimation']['source']}`)"
  )
  lines.append(
    f"- sim timestep: `{kp['sim_timestep']['value']}` "
    f"(source: `{kp['sim_timestep']['source']}`)"
  )
  lines.append(
    f"- control_dt: `{kp['control_dt']['value']}` "
    f"(source: `{kp['control_dt']['source']}`)"
  )
  lines.append(
    f"- action_scale: `{kp['action_scale']['value']}` "
    f"(source: `{kp['action_scale']['source']}`)"
  )
  lines.append(
    f"- policy actuator_names: `{kp['actuator_names_policy']['value']}`"
  )
  lines.append("")
  lines.append("### Actuators (Kp / Kd)")
  lines.append("")
  acts = kp["actuators_Kp_Kd"]["value"]
  if isinstance(acts, list) and acts:
    lines.append("| idx | target_names_expr | Kp | Kd | source |")
    lines.append("|---:|---|---:|---:|---|")
    for row in acts:
      lines.append(
        "| {idx} | `{names}` | {kp} | {kd} | `{src}` |".format(
          idx=row.get("index", "?"),
          names=row.get("target_names_expr", UNRECORDED),
          kp=row.get("stiffness_Kp", UNRECORDED),
          kd=row.get("damping_Kd", UNRECORDED),
          src=row.get("source", UNRECORDED),
        )
      )
  else:
    lines.append(f"- {UNRECORDED}")
  lines.append("")
  lines.append("### PPO / HIM (from agent.yaml)")
  lines.append("")
  for key in (
    "learning_rate",
    "clip_param",
    "gamma",
    "lam",
    "entropy_coef",
    "desired_kl",
    "max_iterations",
    "num_steps_per_env",
    "resume",
    "load_run",
    "load_checkpoint",
  ):
    meta = kp[key]
    lines.append(f"- `{key}` = `{meta['value']}` (source: `{meta['source']}`)")
  him = kp["him"]["value"]
  lines.append(f"- `him` = `{him}` (source: `{kp['him']['source']}`)")
  lines.append("")
  lines.append("### Rewards")
  lines.append("")
  rew = kp["rewards"]["value"]
  items = rew.get("items") if isinstance(rew, dict) else UNRECORDED
  if isinstance(items, dict):
    lines.append("| name | weight | source |")
    lines.append("|---|---:|---|")
    for name, cfg in sorted(items.items()):
      lines.append(
        f"| `{name}` | {cfg.get('weight', UNRECORDED)} | `{cfg.get('source', UNRECORDED)}` |"
      )
  else:
    lines.append(f"- {UNRECORDED}")
  lines.append("")
  lines.append("## 原因推测")
  lines.append("")
  lines.append(
    "- CLI 覆盖参数未单独存档；推测已合并进 `params/*.yaml` "
    "（依据：`mjlab/scripts/train.py` 在 runner 创建前 `dump_yaml` 解析后配置）。"
  )
  lines.append(
    "- Task 名称若显示为 inferred，是因为 run 目录未写入 task_id 字段，"
    "仅根据 `run_name` / 目录名猜测，**不是**已验证结论。"
  )
  if previous_id:
    n_chg = len(diff.get("changed") or [])
    lines.append(
      f"- 与上一归档 `{previous_id}` 相比有 `{n_chg}` 项扁平参数不同；"
      "差异本身是观察，是否影响策略表现需另做验证。"
    )
  else:
    lines.append("- 尚无上一归档可供对比。")
  lines.append("")
  lines.append("## 已验证结论")
  lines.append("")
  lines.append(
    "- 本归档**未**启动训练或仿真；仅复制/解析已有 run 的 params 快照。"
  )
  lines.append(
    "- 缺失字段一律标记为「未记录」，未用当前工作区源码回填历史参数。"
  )
  lines.append(
    f"- 原始 run 目录未被修改；模型与日志仍位于 `{manifest['run_dir']}`。"
  )
  lines.append("")
  lines.append("## Diff vs previous archive")
  lines.append("")
  if previous_id is None:
    lines.append("- (none)")
  else:
    lines.append(f"- Previous archive id: `{previous_id}`")
    lines.append(f"- Changed: `{len(diff.get('changed') or [])}`")
    lines.append(f"- Only in current: `{len(diff.get('only_current') or [])}`")
    lines.append(f"- Only in previous: `{len(diff.get('only_previous') or [])}`")
    for row in (diff.get("changed") or [])[:50]:
      lines.append(
        f"  - `{row['key']}`: `{row['previous']}` → `{row['current']}`"
      )
    if len(diff.get("changed") or []) > 50:
      lines.append("  - … truncated; see `diff_vs_previous.json`")
  lines.append("")
  return "\n".join(lines)


def latest_archive(archives_root: Path) -> Path | None:
  if not archives_root.is_dir():
    return None
  dirs = [p for p in archives_root.iterdir() if p.is_dir()]
  if not dirs:
    return None
  return sorted(dirs, key=lambda p: p.name)[-1]


def archive_train_run(
  run_dir: Path,
  *,
  repo_root: Path,
  archives_root: Path,
  archive_id: str | None = None,
) -> Path:
  run = RunPaths.from_run_dir(run_dir)
  env = load_mjlab_yaml(run.env_yaml)
  agent = load_mjlab_yaml(run.agent_yaml)
  if not isinstance(env, dict) or not isinstance(agent, dict):
    raise ValueError("env.yaml / agent.yaml did not parse to mappings")

  key_params = extract_key_params(env, agent)
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  run_name = run.run_dir.name
  archive_id = archive_id or f"{stamp}__{run_name}"
  archive_dir = _unique_archive_dir(archives_root, archive_id)
  archive_dir.mkdir(parents=True, exist_ok=False)
  (archive_dir / "params").mkdir()
  shutil.copy2(run.env_yaml, archive_dir / "params" / "env.yaml")
  shutil.copy2(run.agent_yaml, archive_dir / "params" / "agent.yaml")

  # Compare against previous archive (exclude the one we just created).
  prev = None
  prev_id = None
  candidates = sorted(
    [p for p in archives_root.iterdir() if p.is_dir() and p != archive_dir],
    key=lambda p: p.name,
  )
  for prev_path in reversed(candidates):
    prev_manifest = prev_path / "manifest.json"
    if not prev_manifest.is_file():
      continue
    prev = _as_key_params_dict(
      json.loads(prev_manifest.read_text(encoding="utf-8")).get("key_params")
    )
    prev_id = prev_path.name
    break

  diff = diff_key_params(key_params, prev)
  if prev_id:
    diff["previous_archive"] = prev_id

  manifest = build_manifest(
    run=run,
    env=env,
    agent=agent,
    repo_root=repo_root,
    archive_dir=archive_dir,
    key_params=key_params,
  )
  (archive_dir / "manifest.json").write_text(
    json.dumps(jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  (archive_dir / "diff_vs_previous.json").write_text(
    json.dumps(jsonable(diff), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  (archive_dir / "report.md").write_text(
    render_report(manifest, diff, prev_id),
    encoding="utf-8",
  )
  write_agent_review(
    archive_dir,
    archive_id=archive_dir.name,
    previous_id=prev_id,
    diff=diff,
  )
  # Pointer back to original run (do not copy large checkpoints).
  (archive_dir / "RUN_DIR.txt").write_text(str(run.run_dir) + "\n", encoding="utf-8")
  return archive_dir


_RUN_DIR_RE = re.compile(
  r"^(?:\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_.*)?)|(?:phase\d+)$"
)


def resolve_run_dir(path: Path, logs_root: Path) -> Path:
  path = path.expanduser()
  if path.is_dir() and (path / "params" / "env.yaml").is_file():
    return path.resolve()
  # Allow shorthand: run folder name under logs/rsl_rl/homie_g1
  cand = logs_root / path.name if not path.is_absolute() else path
  if not cand.is_dir():
    cand = logs_root / str(path)
  if cand.is_dir() and (cand / "params" / "env.yaml").is_file():
    return cand.resolve()
  raise FileNotFoundError(
    f"Cannot resolve training run with params dump: {path}"
  )


# ---------------------------------------------------------------------------
# Play session archive
# ---------------------------------------------------------------------------


def find_train_params_for_checkpoint(
  checkpoint: Path, logs_root: Path
) -> dict[str, Any]:
  """Locate a train run ``params/`` that owns the same checkpoint bytes.

  Returns paths or ``未记录`` — never invents yaml from source.
  """
  checkpoint = checkpoint.resolve()
  result: dict[str, Any] = {
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.is_file() else UNRECORDED,
    "train_run_dir": UNRECORDED,
    "env_yaml": UNRECORDED,
    "agent_yaml": UNRECORDED,
    "match_method": UNRECORDED,
  }
  if not checkpoint.is_file():
    return result

  parent = checkpoint.parent
  if (parent / "params" / "env.yaml").is_file() and (
    parent / "params" / "agent.yaml"
  ).is_file():
    result["train_run_dir"] = str(parent)
    result["env_yaml"] = str(parent / "params" / "env.yaml")
    result["agent_yaml"] = str(parent / "params" / "agent.yaml")
    result["match_method"] = "checkpoint_parent_has_params"
    return result

  # Byte-identical model_*.pt under logs_root with params/
  ref_hash = result["checkpoint_sha256"]
  if ref_hash is UNRECORDED:
    return result
  for pt in logs_root.rglob(checkpoint.name):
    try:
      if pt.resolve() == checkpoint:
        continue
      if not (pt.parent / "params" / "env.yaml").is_file():
        continue
      if file_sha256(pt) == ref_hash:
        result["train_run_dir"] = str(pt.parent.resolve())
        result["env_yaml"] = str(pt.parent / "params" / "env.yaml")
        result["agent_yaml"] = str(pt.parent / "params" / "agent.yaml")
        result["match_method"] = f"sha256_match:{pt}"
        return result
    except OSError:
      continue
  return result


def write_play_meta(
  path: Path,
  *,
  task_id: str,
  checkpoint_file: str | None,
  num_envs: int | None,
  viewer: str | None,
  device: str | None,
  agent_mode: str,
  argv: list[str],
  extra: dict[str, Any] | None = None,
) -> None:
  meta = {
    "schema_version": 1,
    "kind": "play_meta",
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "task_id": task_id,
    "checkpoint_file": checkpoint_file if checkpoint_file else UNRECORDED,
    "num_envs": num_envs if num_envs is not None else UNRECORDED,
    "viewer": viewer if viewer is not None else UNRECORDED,
    "device": device if device is not None else UNRECORDED,
    "agent_mode": agent_mode,
    "argv": argv,
    "source": "explicit play CLI / archive_play_session inputs",
  }
  if extra:
    meta["extra"] = extra
  path.write_text(
    json.dumps(jsonable(meta), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )


def render_play_report(
  manifest: dict[str, Any],
  diff: dict[str, Any],
  previous_id: str | None,
) -> str:
  lines: list[str] = []
  lines.append(f"# Play archive report: `{manifest.get('archive_id', '')}`")
  lines.append("")
  lines.append(f"- Archived at (UTC): `{manifest['archived_at_utc']}`")
  lines.append(f"- Archive dir: `{manifest['archive_dir']}`")
  lines.append("")
  lines.append("## 数据观察")
  lines.append("")
  play = manifest["play"]
  lines.append(f"- Task: `{play.get('task_id', UNRECORDED)}`")
  lines.append(f"- Checkpoint: `{play.get('checkpoint_file', UNRECORDED)}`")
  lines.append(
    f"- Checkpoint sha256: `{play.get('checkpoint_sha256', UNRECORDED)}`"
  )
  lines.append(f"- num_envs: `{play.get('num_envs', UNRECORDED)}`")
  lines.append(f"- viewer: `{play.get('viewer', UNRECORDED)}`")
  lines.append(f"- device: `{play.get('device', UNRECORDED)}`")
  lines.append(f"- agent_mode: `{play.get('agent_mode', UNRECORDED)}`")
  lines.append(f"- argv: `{play.get('argv', UNRECORDED)}`")
  lines.append("")
  train = manifest["train_params_link"]
  lines.append("### Linked train params")
  lines.append("")
  lines.append(f"- match_method: `{train.get('match_method', UNRECORDED)}`")
  lines.append(f"- train_run_dir: `{train.get('train_run_dir', UNRECORDED)}`")
  lines.append(f"- env_yaml: `{train.get('env_yaml', UNRECORDED)}`")
  lines.append(f"- agent_yaml: `{train.get('agent_yaml', UNRECORDED)}`")
  lines.append("")
  lines.append(
    f"- Git HEAD (archive-time): `{manifest['code_version'].get('head', UNRECORDED)}`"
  )
  lines.append(
    f"- Git dirty (archive-time): `{manifest['code_version'].get('dirty', UNRECORDED)}`"
  )
  if manifest.get("key_params") and manifest["key_params"] is not UNRECORDED:
    kp = manifest["key_params"]
    lines.append("")
    lines.append("### Extracted train key params (from linked params only)")
    lines.append("")
    for key in (
      "decimation",
      "sim_timestep",
      "action_scale",
      "seed_agent",
      "learning_rate",
      "clip_param",
      "load_run",
      "load_checkpoint",
    ):
      if key in kp:
        lines.append(
          f"- `{key}` = `{kp[key]['value']}` (source: `{kp[key]['source']}`)"
        )
  else:
    lines.append("")
    lines.append(
      f"- Train key params: `{UNRECORDED}`（checkpoint 旁无 params，且未找到同哈希训练 run）"
    )
  lines.append("")
  lines.append("### Play-resolved config dump")
  lines.append("")
  pr = manifest.get("play_resolved_dump", {})
  lines.append(f"- status: `{pr.get('status', UNRECORDED)}`")
  lines.append(f"- note: `{pr.get('note', UNRECORDED)}`")
  lines.append("")
  lines.append("## 原因推测")
  lines.append("")
  lines.append(
    "- mjlab `play.py` 不会像 train 那样 `dump_yaml`；因此 play 专用覆盖"
    "若未写入 play_meta.json，只能标未记录。"
  )
  if train.get("match_method") == UNRECORDED or train.get("train_run_dir") == UNRECORDED:
    lines.append(
      "- 当前 checkpoint 目录无 params，且未匹配到同字节训练产物时，"
      "训练期 Kp/奖励等必须标未记录，不能用现网源码回填。"
    )
  else:
    lines.append(
      "- 已通过 checkpoint 父目录或 sha256 关联到训练 params；"
      "训练超参观察来自该次 dump，而非当前源码。"
    )
  if previous_id:
    lines.append(
      f"- 与上一归档 `{previous_id}` 的 diff 见下文；差异是否导致行为变化需另验。"
    )
  lines.append("")
  lines.append("## 已验证结论")
  lines.append("")
  lines.append("- 本归档未启动仿真（除非使用带 dump 的 play 包装器另行启动）。")
  lines.append("- 原始 checkpoint / 训练日志未被覆盖。")
  lines.append(
    f"- Play CLI 来源：`{play.get('argv_source', 'explicit inputs')}`。"
  )
  lines.append("")
  lines.append("## Diff vs previous archive")
  lines.append("")
  if previous_id is None:
    lines.append("- (none)")
  else:
    lines.append(f"- Previous: `{previous_id}`")
    lines.append(f"- Changed: `{len(diff.get('changed') or [])}`")
    for row in (diff.get("changed") or [])[:40]:
      lines.append(
        f"  - `{row['key']}`: `{row['previous']}` → `{row['current']}`"
      )
  lines.append("")
  return "\n".join(lines)


def render_agent_review(
  *,
  archive_id: str,
  previous_id: str | None,
  diff: dict[str, Any],
) -> str:
  """Judgment packet. Scripts fill the diff; the agent fills the rest later."""
  changed = diff.get("changed") or []
  lines = [
    "# Agent review",
    "",
    "- status: `pending`",
    f"- archive_id: `{archive_id}`",
    f"- previous_archive: `{previous_id or UNRECORDED}`",
    f"- changed_count: `{len(changed)}`",
    "",
    "## 参数差异",
    "",
    "脚本从相邻归档的 key_params 生成。数字只来自 dump，不要手改。",
    "",
  ]
  if previous_id is None:
    lines.append("- 没有上一份带 key_params 的归档，无法对比。")
  elif not changed:
    lines.append("- 与上一归档的 key_params 无差异，或上一份训练参数为未记录。")
  else:
    for row in changed[:40]:
      lines.append(
        f"- `{row['key']}`: `{row['previous']}` → `{row['current']}`"
      )
    if len(changed) > 40:
      lines.append(f"- … 另有 {len(changed) - 40} 项，见 `diff_vs_previous.json`")
  lines.extend(
    [
      "",
      "## 现象记录",
      "",
      "由用户观察填写。没看到的项写「未记录」，不要根据策略文件名猜测。",
      "",
      "- 弯腰: 未记录",
      "- 站立: 未记录",
      "- 速度跟随: 未记录",
      "- 其他: 未记录",
      "",
      "## 原因推测",
      "",
      "待 agent 填写。只能引用「参数差异」和「现象记录」。",
      "参数或现象为未记录时，对应推测写未记录。",
      "",
      "## 待确认改动",
      "",
      "待 agent 填写具体文件和参数。用户确认前禁止改训练代码、禁止启动训练。",
      "",
      "- 建议: 未记录",
      "- 用户确认: 未确认",
      "",
    ]
  )
  return "\n".join(lines)


def write_agent_review(
  archive_dir: Path,
  *,
  archive_id: str,
  previous_id: str | None,
  diff: dict[str, Any],
) -> Path:
  path = archive_dir / "agent_review.md"
  path.write_text(
    render_agent_review(
      archive_id=archive_id,
      previous_id=previous_id,
      diff=diff,
    ),
    encoding="utf-8",
  )
  return path


def archive_play_session(
  *,
  repo_root: Path,
  archives_root: Path,
  logs_root: Path,
  task_id: str,
  checkpoint_file: Path | None,
  num_envs: int | None,
  viewer: str | None,
  device: str | None,
  agent_mode: str,
  argv: list[str],
  train_run_dir: Path | None = None,
  play_session_dir: Path | None = None,
  archive_id: str | None = None,
  dump_play_resolved: bool = False,
) -> Path:
  """Archive a play session.

  ``dump_play_resolved=True`` will reconstruct play env/agent from *current*
  code and write under ``play_resolved/`` with an explicit warning label.
  Default is False so historical fields stay ``未记录`` when no dump exists.
  """
  archives_root.mkdir(parents=True, exist_ok=True)
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  ckpt_tag = checkpoint_file.stem if checkpoint_file else "no_ckpt"
  archive_id = archive_id or f"{stamp}__play__{ckpt_tag}"
  archive_dir = _unique_archive_dir(archives_root, archive_id)
  archive_dir.mkdir(parents=True, exist_ok=False)

  # Play meta
  if play_session_dir and (play_session_dir / "play_meta.json").is_file():
    shutil.copy2(
      play_session_dir / "play_meta.json", archive_dir / "play_meta.json"
    )
    meta = json.loads(
      (archive_dir / "play_meta.json").read_text(encoding="utf-8")
    )
    task_id = meta.get("task_id", task_id)
    argv = meta.get("argv", argv)
    if meta.get("checkpoint_file") not in (None, UNRECORDED):
      checkpoint_file = Path(str(meta["checkpoint_file"]))
    num_envs = meta.get("num_envs", num_envs)
    if num_envs == UNRECORDED:
      num_envs = None
    viewer = meta.get("viewer", viewer)
    if viewer == UNRECORDED:
      viewer = None
    device = meta.get("device", device)
    if device == UNRECORDED:
      device = None
    agent_mode = meta.get("agent_mode", agent_mode)
  else:
    write_play_meta(
      archive_dir / "play_meta.json",
      task_id=task_id,
      checkpoint_file=str(checkpoint_file) if checkpoint_file else None,
      num_envs=num_envs,
      viewer=viewer,
      device=device,
      agent_mode=agent_mode,
      argv=argv,
    )

  # Train params link
  if train_run_dir is not None:
    train_run_dir = train_run_dir.resolve()
    link = {
      "checkpoint": str(checkpoint_file) if checkpoint_file else UNRECORDED,
      "checkpoint_sha256": (
        file_sha256(checkpoint_file)
        if checkpoint_file and checkpoint_file.is_file()
        else UNRECORDED
      ),
      "train_run_dir": str(train_run_dir),
      "env_yaml": str(train_run_dir / "params" / "env.yaml"),
      "agent_yaml": str(train_run_dir / "params" / "agent.yaml"),
      "match_method": "user_specified_train_run",
    }
    if not (train_run_dir / "params" / "env.yaml").is_file():
      link["env_yaml"] = UNRECORDED
      link["agent_yaml"] = UNRECORDED
      link["match_method"] = "user_specified_train_run_but_params_missing"
  elif checkpoint_file is not None:
    link = find_train_params_for_checkpoint(checkpoint_file, logs_root)
  else:
    link = {
      "checkpoint": UNRECORDED,
      "checkpoint_sha256": UNRECORDED,
      "train_run_dir": UNRECORDED,
      "env_yaml": UNRECORDED,
      "agent_yaml": UNRECORDED,
      "match_method": UNRECORDED,
    }

  key_params: Any = UNRECORDED
  if link.get("env_yaml") not in (None, UNRECORDED) and link.get(
    "agent_yaml"
  ) not in (None, UNRECORDED):
    env_yaml = Path(str(link["env_yaml"]))
    agent_yaml = Path(str(link["agent_yaml"]))
    if env_yaml.is_file() and agent_yaml.is_file():
      params_dir = archive_dir / "train_params"
      params_dir.mkdir()
      shutil.copy2(env_yaml, params_dir / "env.yaml")
      shutil.copy2(agent_yaml, params_dir / "agent.yaml")
      env = load_mjlab_yaml(params_dir / "env.yaml")
      agent = load_mjlab_yaml(params_dir / "agent.yaml")
      if isinstance(env, dict) and isinstance(agent, dict):
        key_params = extract_key_params(env, agent)

  play_resolved = {
    "status": UNRECORDED,
    "note": (
      "mjlab play does not dump resolved cfg. Set dump_play_resolved=True "
      "or use play_with_archive.py to write play_resolved/ explicitly."
    ),
  }
  # Copy pre-dumped play_resolved if present in session dir
  if play_session_dir and (play_session_dir / "play_resolved").is_dir():
    shutil.copytree(
      play_session_dir / "play_resolved", archive_dir / "play_resolved"
    )
    play_resolved = {
      "status": "copied_from_play_session",
      "note": "See play_resolved/; labels inside that dump declare provenance.",
    }
  elif dump_play_resolved:
    play_resolved = _dump_play_resolved_from_current_code(
      archive_dir / "play_resolved",
      task_id=task_id,
      num_envs=num_envs,
    )

  # Diff vs previous archive that has comparable key_params
  prev = None
  prev_id = None
  candidates = sorted(
    [p for p in archives_root.iterdir() if p.is_dir() and p != archive_dir],
    key=lambda p: p.name,
  )
  for prev_path in reversed(candidates):
    prev_manifest_path = prev_path / "manifest.json"
    if not prev_manifest_path.is_file():
      continue
    prev_man = json.loads(prev_manifest_path.read_text(encoding="utf-8"))
    prev = _as_key_params_dict(prev_man.get("key_params"))
    prev_id = prev_path.name
    break
  diff = diff_key_params(
    key_params if isinstance(key_params, dict) else {}, prev
  )
  if prev_id:
    diff["previous_archive"] = prev_id

  manifest = {
    "schema_version": 1,
    "archive_kind": "play",
    "archive_id": archive_id,
    "archived_at_utc": datetime.now(timezone.utc).isoformat(),
    "archive_dir": str(archive_dir),
    "play": {
      "task_id": task_id,
      "checkpoint_file": str(checkpoint_file) if checkpoint_file else UNRECORDED,
      "checkpoint_sha256": link.get("checkpoint_sha256", UNRECORDED),
      "num_envs": num_envs if num_envs is not None else UNRECORDED,
      "viewer": viewer if viewer is not None else UNRECORDED,
      "device": device if device is not None else UNRECORDED,
      "agent_mode": agent_mode,
      "argv": argv,
      "argv_source": "play_meta.json / CLI",
    },
    "train_params_link": link,
    "key_params": key_params if key_params is not UNRECORDED else UNRECORDED,
    "play_resolved_dump": play_resolved,
    "code_version": git_snapshot(repo_root),
    "evidence_classes_note": (
      "Reports must separate 数据观察 / 原因推测 / 已验证结论."
    ),
  }
  (archive_dir / "manifest.json").write_text(
    json.dumps(jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  (archive_dir / "diff_vs_previous.json").write_text(
    json.dumps(jsonable(diff), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  (archive_dir / "report.md").write_text(
    render_play_report(manifest, diff, prev_id),
    encoding="utf-8",
  )
  write_agent_review(
    archive_dir,
    archive_id=archive_dir.name,
    previous_id=prev_id,
    diff=diff,
  )
  return archive_dir


def _dump_play_resolved_from_current_code(
  out_dir: Path,
  *,
  task_id: str,
  num_envs: int | None,
) -> dict[str, Any]:
  """Optional: dump play=True cfg from *current* registry.

  Clearly labeled so it is never confused with historical train params.
  """
  try:
    from dataclasses import asdict

    from mjlab.utils.os import dump_yaml

    # Import package to register tasks.
    import homierl_mjlab  # noqa: F401
    import mjlab.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  except Exception as exc:  # noqa: BLE001
    return {
      "status": UNRECORDED,
      "note": f"Failed to import mjlab/homierl for play dump: {exc}",
    }

  out_dir.mkdir(parents=True, exist_ok=True)
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  if num_envs is not None:
    env_cfg.scene.num_envs = num_envs
  dump_yaml(out_dir / "env.yaml", asdict(env_cfg))
  dump_yaml(out_dir / "agent.yaml", asdict(agent_cfg))
  note = {
    "warning": (
      "THIS IS ARCHIVE-TIME RECONSTRUCTION from current installed code + "
      "play=True, NOT the historical training dump. Do not treat as train "
      "ground truth."
    ),
    "task_id": task_id,
    "num_envs_override": num_envs if num_envs is not None else UNRECORDED,
    "source": "mjlab.tasks.registry.load_env_cfg(task, play=True) at archive time",
  }
  (out_dir / "PROVENANCE.json").write_text(
    json.dumps(jsonable(note), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  return {
    "status": "written_archive_time_reconstruction",
    "note": note["warning"],
    "path": str(out_dir),
  }

# Force-Conditioned ACT under Table Friction Sweep (PushCube-v1)

This documents the project-specific pipeline built on top of the generic ACT baseline (see
[README.md](README.md) for the upstream algorithm/citation) for a specific experiment: does
giving ACT the gripper's contact force (and letting it self-adapt to that signal at test time)
help it generalize across table friction conditions it wasn't necessarily trained to expect?

Three stages: collect demos across a friction/density sweep -> train 3 ACT variants -> evaluate
all 4 comparison cases across the same sweep and plot the result.

## 1. Dataset

### Collecting raw demonstrations

`scripts/data_generation/motionplanning_friction_density.sh` (or its SLURM wrapper,
`motionplanning_friction_density.sbatch`) runs the motion-planning solver on `PushCube-v1` across
7 `(density, static_friction, dynamic_friction)` conditions, collecting 100 successful trajectories
per condition:

```bash
# locally, all 7 conditions sequentially
bash scripts/data_generation/motionplanning_friction_density.sh

# on the cluster, one parallel job per condition
sbatch scripts/data_generation/motionplanning_friction_density.sbatch
```

This writes, per condition:
```
demos_friction/density_<D>_friction_<S>_<Dyn>/PushCube-v1/motionplanning/trajectory.h5
demos_friction/density_<D>_friction_<S>_<Dyn>/PushCube-v1/motionplanning/trajectory.json
```
These raw files are collected with `obs_mode=none` (actions + `env_states` only, no
images/state/force) — deliberately, since motion planning itself doesn't need observations and
recording them during collection would be much slower. They are **not** directly trainable yet.

### Replaying into a training-ready dataset

Before training, replay each condition's raw trajectory through the env with the observations ACT
actually needs (two cameras, proprioception, and `finger_contact_forces`) using `--use-env-states`
so the replayed rollout matches the original motion-planning trajectory exactly:

```bash
for d in demos_friction/*/PushCube-v1/motionplanning; do
  python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "$d/trajectory.h5" \
    -o rgbd --use-env-states --save-traj
done
```

This produces `trajectory.rgbd.pd_joint_pos.physx_cpu.h5` (+ matching `.json`) alongside each raw
file — this is the path you pass as `--demo-path` to training. Each replayed episode contains:

| obs field | shape | notes |
|---|---|---|
| `obs/sensor_data/{base_camera,side_camera}/rgb` | `(T+1, 128, 128, 3)` uint8 | |
| `obs/sensor_data/{base_camera,side_camera}/depth` | `(T+1, 128, 128, 1)` int16 | |
| `obs/agent/qpos`, `qvel` | `(T+1, 9)` | 7 arm + 2 gripper |
| `obs/extra/tcp_pose` | `(T+1, 7)` | pos(3) + quat(4) |
| `obs/extra/finger_contact_forces` | `(T+1, 6)` | left fingertip xyz + right fingertip xyz contact force |
| `actions` | `(T, 8)` | `pd_joint_pos` |

`env_kwargs` in the companion `.json` records the exact `obj_density`/`static_friction`/
`dynamic_friction` used, so each condition's data is traceable even if you rename folders.

## 2. Training

Training turns `finger_contact_forces` into a `direction`(3) + `magnitude`(1) representation and
optionally adds a self-supervised auxiliary head that predicts next-step magnitude from the
current one (see `act/detr/detr_vae.py`'s `ForceMagnitudeEncoder` for the full explanation) —
this auxiliary head is what Test-Time Training adapts at eval time. Three variants are needed
(the eval script's `force_head_no_ttt`/`force_head_ttt` cases reuse the *same* `force_head`
checkpoint, evaluated differently — TTT is an eval-time-only behavior):

| variant | flags | used for eval case(s) |
|---|---|---|
| `no_force` | `--no-include-force` | ACT w/o force |
| `force` | `--include-force --no-use-force-magnitude-head` | ACT w/ force, w/o head |
| `force_head` | `--include-force --use-force-magnitude-head` | ACT w/ force+head, w/ and w/o TTT |

### Command line (one variant)

```bash
python train_rgbd.py \
  --env-id PushCube-v1 \
  --demo-path demos_friction/density_1000_friction_3.3_2.3/PushCube-v1/motionplanning/trajectory.rgbd.pd_joint_pos.physx_cpu.h5 \
  --control-mode pd_joint_pos \
  --exp-name act_force_head \
  --include-force --use-force-magnitude-head \
  --save-freq 20000
```
(`--save-freq` guarantees a checkpoint exists even if the policy never beats a previous best
success rate during training-time periodic eval, which otherwise is the only thing that triggers
a checkpoint save.) Checkpoints land in `runs/<exp-name>/checkpoints/`.

Which `--demo-path` you point at determines which friction/density condition the *training* data
comes from — the friction sweep is evaluated at test time (see below), not trained on across
multiple conditions here. If you want the policy to have seen multiple conditions during training
too, you'd need to merge multiple conditions' replayed trajectory files first
(`mani_skill.trajectory.merge_trajectory`) — not done by the scripts here by default.

### sbatch (all 3 variants as a job array)

```bash
DEMO_PATH=demos_friction/density_1000_friction_3.3_2.3/PushCube-v1/motionplanning/trajectory.rgbd.pd_joint_pos.physx_cpu.h5 \
  sbatch train_friction_comparison.sbatch
```
`--array=0-2` maps to `no_force` / `force` / `force_head` respectively. Edit `DEMO_PATH` in the
script (or override via the env var above) to point at your actual replayed dataset — it currently
defaults to the single-condition smoke-test file used while building this pipeline.

## 3. Evaluation

`eval_friction_sweep.py` loads up to 3 checkpoints, runs each of the resulting 4 cases across all
7 friction/density conditions, and records success rate + mean (normalized dense) return per
condition:

| case | checkpoint | eval-time behavior |
|---|---|---|
| `no_force` | `--ckpt-no-force` | frozen, no force input at all |
| `force` | `--ckpt-force` | frozen, force input but no magnitude head |
| `force_head_no_ttt` | `--ckpt-force-head` | frozen, force + magnitude head (head unused for adaptation) |
| `force_head_ttt` | `--ckpt-force-head` | **online adaptation**: every rollout step, take a gradient step on `ForceMagnitudeEncoder` + the magnitude head only (self-supervised next-magnitude-prediction loss), using the just-observed true magnitude as the label. Action head / vision backbone / everything else stays frozen. |

Pass only the checkpoints you have — any case whose checkpoint arg is omitted is skipped.

### Command line

```bash
python eval_friction_sweep.py \
  --env-id PushCube-v1 \
  --control-mode pd_joint_pos \
  --sim-backend physx_cuda \
  --num-eval-episodes 100 \
  --num-eval-envs 25 \
  --ckpt-no-force runs/act_no_force/checkpoints/best_eval_success_once.pt \
  --ckpt-force runs/act_force/checkpoints/best_eval_success_once.pt \
  --ckpt-force-head runs/act_force_head/checkpoints/best_eval_success_once.pt \
  --output-dir runs/friction_sweep_eval
```
Architecture flags (`--hidden-dim`, `--nheads`, etc.) default to match `train_rgbd.py`'s own
defaults — pass matching overrides if you changed them at training time, since they aren't saved
in the checkpoint itself.

Use `--sim-backend physx_cuda` (GPU) rather than CPU for real runs: the CPU backend's multi-env
path has a pre-existing bug unrelated to this pipeline (depth observations are `float16`, which
Python's multiprocessing shared memory can't represent), and GPU is faster anyway for a
4-case x 7-condition x 100-episode sweep.

### sbatch (all cases + full sweep in one job)

```bash
CKPT_NO_FORCE=runs/act_no_force/checkpoints/best_eval_success_once.pt \
CKPT_FORCE=runs/act_force/checkpoints/best_eval_success_once.pt \
CKPT_FORCE_HEAD=runs/act_force_head/checkpoints/best_eval_success_once.pt \
  sbatch eval_friction_comparison.sbatch
```
Checkpoint filenames depend on when/whether a training run hit a new best success rate
(`best_eval_success_once.pt`/`best_eval_success_at_end.pt`) or on `--save-freq` (numbered by
iteration, e.g. `20000.pt`) — check `runs/<exp-name>/checkpoints/` after training to see what
actually got saved before filling these in.

### Output

- `runs/friction_sweep_eval/friction_sweep_results.json` — raw `{case: {condition: {success_rate, mean_return, ...}}}`.
- `runs/friction_sweep_eval/friction_sweep_comparison.png` — two panels (success rate and return,
  each vs. static friction), one line per case run.

"""Render side-by-side comparison videos for demo episodes that exist in both the raw
pd_joint_pos motion-planning trajectory file and its pd_ee_pose control-mode conversion.

trajectory.h5 (600 episodes, keys traj_0..traj_599) and
trajectory.rgbd.pd_ee_pose.physx_cpu.h5 (122 episodes, a subset of the same keys -- control-mode
conversion is lossy, so only ~20% of episodes survive) share episode ids by construction: episode
`traj_N` in the converted file is the pd_ee_pose replay of `traj_N` in the raw file. This script:

1. Finds the episode ids present in the pd_ee_pose file.
2. Extracts those same episodes from both files into two small subset trajectory files (same
   h5-group-copy technique used by mani_skill.trajectory.merge_trajectory), in matching sorted
   order.
3. Replays each subset with `mani_skill.trajectory.replay_trajectory --use-env-states --save-video`
   (exact reproduction, no action re-simulation) to render one video per episode.
4. Combines each matching pair into one side-by-side video, freeze-padding whichever side is
   shorter so length differences (e.g. the inflated elapsed_steps seen in some pd_ee_pose
   conversions) stay visible instead of being cut off.

Usage:
    python scripts/data_generation/compare_pd_joint_vs_ee_pose_videos.py
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import h5py
import imageio_ffmpeg

from mani_skill.utils.io_utils import dump_json, load_json

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def build_subset(src_h5_path: str, episode_ids: list, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5_path = out_dir / "trajectory.h5"
    src_json = load_json(str(src_h5_path).replace(".h5", ".json"))
    episodes_by_id = {ep["episode_id"]: ep for ep in src_json["episodes"]}

    out_json = {k: v for k, v in src_json.items() if k != "episodes"}
    out_json["episodes"] = [episodes_by_id[eid] for eid in episode_ids]

    with h5py.File(src_h5_path, "r") as src_h5, h5py.File(out_h5_path, "w") as out_h5:
        for eid in episode_ids:
            traj_id = f"traj_{eid}"
            src_h5.copy(traj_id, out_h5, traj_id)

    dump_json(str(out_h5_path).replace(".h5", ".json"), out_json, indent=2)
    return out_h5_path


def replay_and_render(subset_h5_path: Path, sim_backend: str):
    cmd = [
        sys.executable, "-m", "mani_skill.trajectory.replay_trajectory",
        "--traj-path", str(subset_h5_path),
        "--use-env-states",
        "--save-video",
        "--allow-failure",  # keep every episode so video counts/order stay 1:1 across both
        # subsets -- without this, a failed replay is silently dropped and every subsequent
        # video index shifts by one, pairing the wrong episodes together in the combine step.
        "-b", sim_backend,
    ]
    print(f"[replay] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def probe_duration(video_path: Path) -> float:
    # imageio_ffmpeg only bundles ffmpeg, not ffprobe, so parse the "Duration: HH:MM:SS.ss" line
    # ffmpeg prints on stderr when given no output.
    out = subprocess.run(
        [FFMPEG, "-i", str(video_path)], capture_output=True, text=True,
    )
    match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", out.stderr)
    if not match:
        raise RuntimeError(f"Could not parse duration for {video_path}:\n{out.stderr}")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def combine_side_by_side(left: Path, right: Path, out_path: Path):
    # NOTE: the ffmpeg binary bundled by imageio_ffmpeg has no drawtext filter, so labels aren't
    # burned into the frame -- left is always pd_joint_pos, right is always pd_ee_pose (see the
    # output filename, which spells this out per pair).
    dur_l, dur_r = probe_duration(left), probe_duration(right)
    pad_l = max(0.0, dur_r - dur_l)
    pad_r = max(0.0, dur_l - dur_r)
    filter_complex = (
        f"[0:v]tpad=stop_mode=clone:stop_duration={pad_l}[l];"
        f"[1:v]tpad=stop_mode=clone:stop_duration={pad_r}[r];"
        f"[l][r]hstack=inputs=2[v]"
    )
    cmd = [
        FFMPEG, "-y", "-i", str(left), "-i", str(right),
        "-filter_complex", filter_complex,
        "-map", "[v]", "-c:v", "libx264", "-crf", "20", str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    demo_dir = "demos_friction/density_1000_friction_3.3_2.3/PushCube-v1/motionplanning"
    parser.add_argument("--pd-joint-pos-path", default=f"{demo_dir}/trajectory.h5")
    parser.add_argument("--pd-ee-pose-path", default=f"{demo_dir}/trajectory.rgbd.pd_ee_pose.physx_cpu.h5")
    parser.add_argument("--output-dir", default=f"{demo_dir}/video_compare")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--skip-replay", action="store_true",
                         help="Skip subset-building/replay and only (re-)combine videos already rendered by a prior run.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    joint_pos_dir = out_dir / "pd_joint_pos_subset"
    ee_pose_dir = out_dir / "pd_ee_pose_subset"
    combined_dir = out_dir / "comparisons"
    combined_dir.mkdir(parents=True, exist_ok=True)

    ee_pose_json = load_json(str(args.pd_ee_pose_path).replace(".h5", ".json"))
    episode_ids = sorted(ep["episode_id"] for ep in ee_pose_json["episodes"])
    print(f"Found {len(episode_ids)} matching episodes between the two files.")

    if not args.skip_replay:
        joint_pos_subset = build_subset(args.pd_joint_pos_path, episode_ids, joint_pos_dir)
        ee_pose_subset = build_subset(args.pd_ee_pose_path, episode_ids, ee_pose_dir)

        replay_and_render(joint_pos_subset, args.sim_backend)
        replay_and_render(ee_pose_subset, args.sim_backend)

    ok, failed = 0, []
    for i, eid in enumerate(episode_ids):
        left = joint_pos_dir / f"{i}.mp4"
        right = ee_pose_dir / f"{i}.mp4"
        if not left.exists() or not right.exists():
            failed.append(eid)
            continue
        out_path = combined_dir / f"traj_{eid}_pd_joint_pos_vs_pd_ee_pose.mp4"
        combine_side_by_side(left, right, out_path)
        ok += 1

    print(f"Combined {ok}/{len(episode_ids)} pairs into {combined_dir}")
    if failed:
        print(f"Missing rendered video(s) for episode ids: {failed}")


if __name__ == "__main__":
    main()

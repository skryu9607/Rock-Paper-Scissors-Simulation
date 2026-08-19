#!/usr/bin/env python3
"""Generate large baseline datasets from alpha_simulator.py.

This script intentionally reuses the existing microscopic dynamics instead of
changing them.  It records enough information to reconstruct later
connected-groups / active-frontiers offline.

Outputs
-------
<out>/summary.csv
    One row per episode: seed, sampled/fixed initial population, first-extinction time,
    terminal time, winner, timeout flag, and final population.

<out>/baseline_XXXXX_YYYYY.h5
    Sharded microscopic trajectories.  Each episode contains an explicit
    initial_configuration subgroup (all agent identities, types, positions,
    velocities, headings, and speeds) plus sampled trajectory steps.

Type encoding in HDF5: R=0, S=1, P=2.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np

from alpha_simulator import (
    Agent,
    BASE_SPEED,
    COLLISION_RADIUS,
    HEIGHT,
    IMG_HEIGHT,
    IMG_WIDTH,
    PREY_OF,
    SENSE_RANGE,
    TURN_RATE,
    WIDTH,
    collide_pair,
    get_population,
)

TYPE_TO_INT = {"R": 0, "S": 1, "P": 2}
INT_TO_TYPE = {0: "R", 1: "S", 2: "P"}


@dataclass
class EpisodeSummary:
    episode_id: int
    seed: int
    init_R: int
    init_S: int
    init_P: int
    first_extinction_step: int
    terminal_step: int
    winner: str
    timed_out: int
    final_R: int
    final_S: int
    final_P: int
    n_snapshots: int


def _alive_types(pop: Sequence[int]) -> int:
    return sum(int(n > 0) for n in pop)


def _sample_initial_population(
    episode_id: int,
    seed_base: int,
    mode: str,
    fixed_initial: Tuple[int, int, int],
    total_agents: int,
    min_per_type: int,
) -> Tuple[int, int, int]:
    """Deterministically choose the initial R/S/P composition for one episode.

    simplex mode samples *uniformly over feasible integer compositions*, rather
    than multinomially. This deliberately covers imbalanced as well as balanced
    populations instead of concentrating near N/3,N/3,N/3.
    """
    if mode == "fixed":
        return fixed_initial

    if total_agents < 3 * min_per_type:
        raise ValueError("total_agents must be >= 3 * min_per_type")

    rng = random.Random(seed_base + 1_000_003 * episode_id + 17)
    feasible = []
    lo = min_per_type
    hi = total_agents - 2 * min_per_type
    for r in range(lo, hi + 1):
        for s in range(lo, total_agents - r - min_per_type + 1):
            p = total_agents - r - s
            if p >= min_per_type:
                feasible.append((r, s, p))
    return feasible[rng.randrange(len(feasible))]


def _make_agents(initial_pop: Tuple[int, int, int], seed: int) -> List[Agent]:
    random.seed(seed)
    n_R, n_S, n_P = initial_pop
    agents: List[Agent] = []
    for agent_type, count in zip(["R", "S", "P"], [n_R, n_S, n_P]):
        for _ in range(count):
            x = random.uniform(0, WIDTH - IMG_WIDTH)
            y = random.uniform(0, HEIGHT - IMG_HEIGHT)
            agents.append(Agent(x, y, agent_type))
    return agents


def _snapshot(agents):
    n = len(agents)
    pos = np.empty((n, 2), dtype=np.float32)
    vel = np.empty((n, 2), dtype=np.float32)
    typ = np.empty((n,), dtype=np.uint8)
    for i, a in enumerate(agents):
        pos[i, 0] = a.cx
        pos[i, 1] = a.cy
        # Exactly matches Agent.move(): x += sin(angle)*speed, y -= cos(angle)*speed
        vel[i, 0] = math.sin(a.angle) * a.speed
        vel[i, 1] = -math.cos(a.angle) * a.speed
        typ[i] = TYPE_TO_INT[a.agent_type]
    pop = np.asarray(get_population(agents), dtype=np.uint16)
    return pos, vel, typ, pop


def _initial_configuration(agents: Sequence[Agent]):
    """Capture the complete microscopic initial state needed for analysis/replay.

    We save angle and speed explicitly even though velocity is also saved.  This
    avoids relying on inverse reconstruction and makes the baseline artifact
    self-contained if the simulator dynamics are later changed.
    """
    pos, vel, typ, pop = _snapshot(agents)
    n = len(agents)
    angle = np.asarray([a.angle for a in agents], dtype=np.float64)
    speed = np.asarray([a.speed for a in agents], dtype=np.float32)
    agent_id = np.arange(n, dtype=np.int32)
    return {
        "agent_id": agent_id,
        "type": typ.copy(),
        "position_xy": pos.copy(),
        "velocity_xy": vel.copy(),
        "angle_rad": angle,
        "speed": speed,
        "population_RSP": pop.copy(),
    }


def simulate_episode(
    episode_id: int,
    seed: int,
    initial_pop: Tuple[int, int, int],
    policy: str,
    max_steps: int,
    snapshot_stride: int,
):
    """Run until one type remains or max_steps is reached.

    Note: the existing alpha_simulator.run_episode() stops at the *first*
    extinction.  For baseline outcome labels we need both:
      - first_extinction_step = coexistence duration
      - winner/terminal_step  = eventual single surviving type (if reached)
    """
    agents = _make_agents(initial_pop, seed)
    initial_configuration = _initial_configuration(agents)

    steps: List[int] = []
    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    types: List[np.ndarray] = []
    populations: List[np.ndarray] = []

    def record(step: int):
        pos, vel, typ, pop = _snapshot(agents)
        steps.append(step)
        positions.append(pos)
        velocities.append(vel)
        types.append(typ)
        populations.append(pop)

    record(0)
    first_extinction_step = -1
    terminal_step = max_steps
    winner = ""
    timed_out = 1
    last_recorded_step = 0

    for step in range(1, max_steps + 1):
        # Preserve alpha_simulator.py policy semantics.
        if policy != "random":
            actions = []
            for agent in agents:
                obs = agent.observe(agents)
                action = agent.select_action(obs, policy)
                actions.append((agent, action, obs))
            for agent, action, obs in actions:
                agent.apply_action(action, obs)

        for agent in agents:
            agent.move()

        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                collide_pair(agents[i], agents[j])

        pop = get_population(agents)
        alive = _alive_types(pop)

        just_reached_first_extinction = alive < 3 and first_extinction_step < 0
        if just_reached_first_extinction:
            first_extinction_step = step

        must_record = (
            step % snapshot_stride == 0
            or just_reached_first_extinction
            or alive == 1
            or step == max_steps
        )
        if must_record and step != last_recorded_step:
            record(step)
            last_recorded_step = step

        if alive == 1:
            terminal_step = step
            timed_out = 0
            if pop[0] > 0:
                winner = "R"
            elif pop[1] > 0:
                winner = "S"
            else:
                winner = "P"
            break

    if first_extinction_step < 0:
        first_extinction_step = max_steps

    # Ensure terminal/max step is represented.
    if steps[-1] != terminal_step:
        record(terminal_step)

    final_pop = tuple(int(x) for x in populations[-1])
    summary = EpisodeSummary(
        episode_id=episode_id,
        seed=seed,
        init_R=initial_pop[0],
        init_S=initial_pop[1],
        init_P=initial_pop[2],
        first_extinction_step=first_extinction_step,
        terminal_step=terminal_step,
        winner=winner if winner else "TIMEOUT",
        timed_out=timed_out,
        final_R=final_pop[0],
        final_S=final_pop[1],
        final_P=final_pop[2],
        n_snapshots=len(steps),
    )

    trajectory = {
        "initial_configuration": initial_configuration,
        "steps": np.asarray(steps, dtype=np.int32),
        "population": np.stack(populations, axis=0),
        "position": np.stack(positions, axis=0),
        "velocity": np.stack(velocities, axis=0),
        "type": np.stack(types, axis=0),
    }
    return summary, trajectory


def _compression_kwargs(name: str):
    if name == "none":
        return {}
    if name == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    if name == "lzf":
        return {"compression": "lzf", "shuffle": True}
    raise ValueError(name)


def generate_shard(args_tuple):
    (
        shard_id,
        episode_ids,
        output_dir,
        initial_mode,
        fixed_initial,
        total_agents,
        min_per_type,
        policy,
        max_steps,
        snapshot_stride,
        seed_base,
        compression,
    ) = args_tuple

    start_id, end_id = episode_ids[0], episode_ids[-1]
    shard_path = Path(output_dir) / f"baseline_{start_id:05d}_{end_id:05d}.h5"
    summaries = []
    ck = _compression_kwargs(compression)

    with h5py.File(shard_path, "w") as h5:
        # Reproducibility metadata: freeze the microscopic model parameters.
        h5.attrs["engine"] = "alpha_simulator.py"
        h5.attrs["policy"] = policy
        h5.attrs["initial_mode"] = initial_mode
        if initial_mode == "fixed":
            h5.attrs["initial_population_RSP"] = fixed_initial
        else:
            h5.attrs["total_agents"] = total_agents
            h5.attrs["min_per_type"] = min_per_type
        h5.attrs["max_steps"] = max_steps
        h5.attrs["snapshot_stride"] = snapshot_stride
        h5.attrs["type_encoding"] = "R=0,S=1,P=2"
        h5.attrs["WIDTH"] = WIDTH
        h5.attrs["HEIGHT"] = HEIGHT
        h5.attrs["IMG_WIDTH"] = IMG_WIDTH
        h5.attrs["IMG_HEIGHT"] = IMG_HEIGHT
        h5.attrs["COLLISION_RADIUS"] = COLLISION_RADIUS
        h5.attrs["BASE_SPEED"] = BASE_SPEED
        h5.attrs["SENSE_RANGE"] = SENSE_RANGE
        h5.attrs["TURN_RATE"] = TURN_RATE
        h5.attrs["cyclic_dominance"] = "R>S, S>P, P>R"
        h5.attrs["note"] = (
            "Agents are never deleted; collision converts loser type to winner type. "
            "first_extinction_step marks end of 3-type coexistence; terminal_step/winner "
            "requires a single remaining type. Every episode also stores an explicit "
            "initial_configuration subgroup for prediction-vs-outcome comparisons."
        )

        for ep_id in episode_ids:
            seed = seed_base + ep_id
            initial_pop = _sample_initial_population(
                ep_id, seed_base, initial_mode, fixed_initial, total_agents, min_per_type
            )
            summary, tr = simulate_episode(
                ep_id, seed, initial_pop, policy, max_steps, snapshot_stride
            )
            summaries.append(asdict(summary))

            g = h5.create_group(f"episode_{ep_id:05d}")
            for k, v in asdict(summary).items():
                g.attrs[k] = v

            # Explicit immutable initial microstate.  Do not rely only on
            # trajectory index 0: this subgroup is the canonical input state
            # for future winner/coexistence prediction experiments.
            ig = g.create_group("initial_configuration")
            ig.attrs["step"] = 0
            ig.attrs["seed"] = seed
            ig.attrs["coordinate_definition"] = "agent center (cx, cy)"
            ig.create_dataset("agent_id", data=tr["initial_configuration"]["agent_id"], **ck)
            ig.create_dataset("type", data=tr["initial_configuration"]["type"], **ck)
            ig.create_dataset("position_xy", data=tr["initial_configuration"]["position_xy"], **ck)
            ig.create_dataset("velocity_xy", data=tr["initial_configuration"]["velocity_xy"], **ck)
            ig.create_dataset("angle_rad", data=tr["initial_configuration"]["angle_rad"], **ck)
            ig.create_dataset("speed", data=tr["initial_configuration"]["speed"], **ck)
            ig.create_dataset("population_RSP", data=tr["initial_configuration"]["population_RSP"], **ck)

            g.create_dataset("step", data=tr["steps"], **ck)
            g.create_dataset("population_RSP", data=tr["population"], **ck)
            g.create_dataset("position_xy", data=tr["position"], **ck)
            g.create_dataset("velocity_xy", data=tr["velocity"], **ck)
            g.create_dataset("type", data=tr["type"], **ck)

    return shard_id, str(shard_path), summaries


def write_summary_csv(path: Path, rows: List[dict]):
    rows = sorted(rows, key=lambda r: r["episode_id"])
    fieldnames = list(EpisodeSummary.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument(
        "--initial",
        nargs=3,
        type=int,
        metavar=("R", "S", "P"),
        default=(4, 4, 4),
        help="Initial R S P counts (default: 4 4 4)",
    )
    p.add_argument(
        "--initial-mode",
        choices=["fixed", "simplex"],
        default="fixed",
        help=("fixed: use --initial for every episode; simplex: uniformly sample "
              "integer R/S/P compositions with fixed --total-agents and --min-per-type"),
    )
    p.add_argument(
        "--total-agents",
        type=int,
        default=60,
        help="Total population used by --initial-mode simplex (default: 60)",
    )
    p.add_argument(
        "--min-per-type",
        type=int,
        default=5,
        help="Minimum initial count of each type in simplex mode (default: 5)",
    )
    p.add_argument(
        "--policy",
        choices=["random", "aggressive", "defensive", "balanced"],
        default="random",
    )
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument(
        "--snapshot-stride",
        type=int,
        default=1,
        help="Record every k simulation steps; terminal/extinction steps are always recorded.",
    )
    p.add_argument("--seed-base", type=int, default=100_000)
    p.add_argument("--shard-size", type=int, default=250)
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) - 1)),
    )
    p.add_argument("--compression", choices=["lzf", "gzip", "none"], default="lzf")
    p.add_argument("--output", default="baseline_dataset")
    return p.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive")
    if args.snapshot_stride <= 0:
        raise SystemExit("--snapshot-stride must be positive")
    if any(n < 0 for n in args.initial) or sum(args.initial) == 0:
        raise SystemExit("--initial counts must be nonnegative and total > 0")
    if args.initial_mode == "simplex":
        if args.min_per_type < 1:
            raise SystemExit("--min-per-type must be >= 1 for three-type coexistence data")
        if args.total_agents < 3 * args.min_per_type:
            raise SystemExit("--total-agents must be >= 3 * --min-per-type")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_ids = list(range(args.episodes))
    shards = [
        episode_ids[i : i + args.shard_size]
        for i in range(0, len(episode_ids), args.shard_size)
    ]

    tasks = []
    for shard_id, ids in enumerate(shards):
        tasks.append(
            (
                shard_id,
                ids,
                str(output_dir),
                args.initial_mode,
                tuple(args.initial),
                args.total_agents,
                args.min_per_type,
                args.policy,
                args.max_steps,
                args.snapshot_stride,
                args.seed_base,
                args.compression,
            )
        )

    print("=" * 72)
    print("RPS BASELINE DATASET GENERATOR")
    print(f"episodes        : {args.episodes}")
    print(f"initial mode    : {args.initial_mode}")
    if args.initial_mode == "fixed":
        print(f"initial R,S,P   : {tuple(args.initial)}")
    else:
        print(f"total agents    : {args.total_agents}")
        print(f"min per type    : {args.min_per_type}")
    print(f"policy          : {args.policy}")
    print(f"max_steps       : {args.max_steps}")
    print(f"snapshot_stride : {args.snapshot_stride}")
    print(f"workers         : {args.workers}")
    print(f"shards          : {len(shards)} x up to {args.shard_size} episodes")
    print(f"output          : {output_dir.resolve()}")
    print("=" * 72)

    all_rows: List[dict] = []
    completed = 0

    if args.workers == 1:
        results = map(generate_shard, tasks)
        for shard_id, shard_path, rows in results:
            all_rows.extend(rows)
            completed += len(rows)
            print(f"[{completed:>6}/{args.episodes}] wrote {shard_path}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(generate_shard, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                shard_id, shard_path, rows = fut.result()
                all_rows.extend(rows)
                completed += len(rows)
                print(f"[{completed:>6}/{args.episodes}] wrote {shard_path}")

    summary_path = output_dir / "summary.csv"
    write_summary_csv(summary_path, all_rows)

    # Small text manifest for reproducibility.
    winner_counts = {"R": 0, "S": 0, "P": 0, "TIMEOUT": 0}
    for r in all_rows:
        winner_counts[r["winner"]] = winner_counts.get(r["winner"], 0) + 1
    manifest = output_dir / "manifest.txt"
    with manifest.open("w") as f:
        f.write(f"episodes={args.episodes}\n")
        f.write(f"initial_mode={args.initial_mode}\n")
        if args.initial_mode == "fixed":
            f.write(f"initial_RSP={tuple(args.initial)}\n")
        else:
            f.write(f"total_agents={args.total_agents}\n")
            f.write(f"min_per_type={args.min_per_type}\n")
        f.write(f"policy={args.policy}\n")
        f.write(f"max_steps={args.max_steps}\n")
        f.write(f"snapshot_stride={args.snapshot_stride}\n")
        f.write(f"seed_base={args.seed_base}\n")
        f.write(f"winner_counts={winner_counts}\n")

    print("\nDone.")
    print(f"Summary : {summary_path}")
    print(f"Manifest: {manifest}")
    print(f"Winners : {winner_counts}")


if __name__ == "__main__":
    main()

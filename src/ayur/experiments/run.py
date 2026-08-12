"""Staged experiment runner: smoke -> pilot -> structured -> llm-subset.

Laptop-safe by construction:
  * appends each trajectory to JSONL and flushes every CHECKPOINT_EVERY cases
  * resumes from the last completed case, recomputing nothing
  * reports elapsed and estimated-remaining time continuously
  * watches resident memory and stops gracefully before the machine suffers
  * handles SIGINT/SIGTERM by finishing the current case and saving

Nothing is silently truncated: every run writes its exact sample size into the
manifest, and a run that stops early says so.
"""
from __future__ import annotations

import argparse
import json
import resource
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ayur.data.prep import build_condition_space
from ayur.env.patient import PatientEnvironment
from ayur.env_detect import detect
from ayur.experiments.agents import build_agents

RESULTS = Path("results")
CHECKPOINTS = RESULTS / "checkpoints"
CHECKPOINT_EVERY = 25

#: Stop if resident memory exceeds this fraction of unified memory.
MEMORY_STOP_FRACTION = 0.65


@dataclass
class StageConfig:
    name: str
    n_cases: int | None          # None = every case the stage can generate
    max_questions: int
    tau: float
    noise: float                 # the environment's true generative noise
    omission_rate: float
    seed: int = 0
    #: What the agent's posterior *assumes* the noise is. When this differs from
    #: `noise` the likelihood is misspecified - which is the realistic case and
    #: the one the paper should headline. None means well-specified.
    assumed_noise: float | None = None

    @property
    def agent_noise(self) -> float:
        return self.noise if self.assumed_noise is None else self.assumed_noise

    @property
    def misspecified(self) -> bool:
        return self.assumed_noise is not None and self.assumed_noise != self.noise


STAGES = {
    # 10-20 cases, all non-LLM components, completes in seconds.
    "smoke": StageConfig("smoke", 20, 15, 0.5, 0.05, 0.1),
    # 100-300 cases, all principal systems, timed to project the full run.
    "pilot": StageConfig("pilot", 200, 15, 0.5, 0.05, 0.1),
    # Full structured evaluation - no LLM inference anywhere in this stage.
    "structured": StageConfig("structured", 5000, 20, 0.5, 0.05, 0.1),
    # The honest headline: the agent's likelihood is wrong about the noise and
    # the patient omits more than it expects. Absolute numbers drop; the point
    # is whether the policy ranking survives.
    "structured-misspecified": StageConfig(
        "structured-misspecified", 5000, 20, 0.5,
        noise=0.15, omission_rate=0.25, assumed_noise=0.05,
    ),
    # Reserved for the LLM baselines (B6); bounded and resumable.
    "llm-subset": StageConfig("llm-subset", 200, 15, 0.5, 0.05, 0.1),
}


def rss_gb() -> float:
    """Resident set size. On macOS ru_maxrss is bytes; on Linux it is KiB."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024**3 if sys.platform == "darwin" else r / 1024**2


def fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


class GracefulStop:
    """Finish the case in flight, then stop. Never lose completed work."""

    def __init__(self):
        self.requested = False
        signal.signal(signal.SIGINT, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        if self.requested:
            print("\n  second interrupt - exiting immediately", flush=True)
            raise KeyboardInterrupt
        self.requested = True
        print(
            f"\n  interrupt received ({signum}); finishing current case and saving. "
            "Press Ctrl-C again to abort.",
            flush=True,
        )


def completed_keys(path: Path) -> set[tuple[int, str]]:
    """(case_index, agent) pairs already present, so a resume skips them."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A partial final line from a hard kill; ignore it.
                continue
            done.add((rec["case_index"], rec["agent"]))
    return done


def run_stage(stage: str, resume: bool = True, override_n: int | None = None) -> dict:
    cfg = STAGES[stage]
    if override_n is not None:
        cfg = StageConfig(**{**asdict(cfg), "n_cases": override_n})

    env_info = detect()
    memory_limit = (env_info.memory_gb or 16) * MEMORY_STOP_FRACTION

    space = build_condition_space()
    patient_env = PatientEnvironment(
        space, noise=cfg.noise, omission_rate=cfg.omission_rate, seed=cfg.seed
    )
    agents = build_agents(
        max_questions=cfg.max_questions, tau=cfg.tau, noise=cfg.agent_noise
    )

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINTS / f"{stage}_trajectories.jsonl"
    done = completed_keys(out_path) if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()

    total = cfg.n_cases * len(agents)
    print("=" * 70)
    print(f"STAGE: {stage}")
    print("=" * 70)
    print(f"  conditions        {space.n_conditions}")
    print(f"  features          {space.n_features}")
    print(f"  cases             {cfg.n_cases}")
    print(f"  agents            {len(agents)}  ({', '.join(a.name for a in agents)})")
    print(f"  total runs        {total}")
    print(f"  env noise         {cfg.noise}   omission {cfg.omission_rate}")
    print(f"  agent assumes     {cfg.agent_noise}"
          f"{'   ** MISSPECIFIED **' if cfg.misspecified else '   (well-specified)'}")
    print(f"  max questions     {cfg.max_questions}   tau {cfg.tau}")
    if done:
        print(f"  RESUMING          {len(done)} runs already complete")
    print(f"  memory guard      stop above {memory_limit:.1f} GB RSS")
    print("-" * 70)

    stopper = GracefulStop()
    t0 = time.perf_counter()
    n_done = len(done)
    n_new = 0
    stopped_early = False
    reason = "completed"

    with out_path.open("a") as fh:
        for case in patient_env.cases(cfg.n_cases):
            if stopper.requested:
                stopped_early, reason = True, "interrupted"
                break
            if rss_gb() > memory_limit:
                stopped_early, reason = True, f"memory guard ({rss_gb():.1f} GB)"
                print(f"\n  ! stopping: RSS {rss_gb():.1f} GB exceeds {memory_limit:.1f} GB")
                break

            for agent in agents:
                if (case.index, agent.name) in done:
                    continue
                traj = agent.run(case, patient_env, space.matrix)
                fh.write(json.dumps(traj.to_json()) + "\n")
                n_done += 1
                n_new += 1

            if n_new and (case.index + 1) % CHECKPOINT_EVERY == 0:
                fh.flush()
                elapsed = time.perf_counter() - t0
                rate = n_new / elapsed if elapsed else 0
                remaining = (total - n_done) / rate if rate else 0
                print(
                    f"  case {case.index + 1:>6}/{cfg.n_cases}  "
                    f"runs {n_done}/{total}  "
                    f"elapsed {fmt_hms(elapsed)}  eta {fmt_hms(remaining)}  "
                    f"rss {rss_gb():.2f} GB",
                    flush=True,
                )
        fh.flush()

    elapsed = time.perf_counter() - t0
    manifest = {
        "stage": stage,
        "config": asdict(cfg),
        "n_conditions": space.n_conditions,
        "n_features": space.n_features,
        "keep_min": space.keep_min,
        "agents": [a.name for a in agents],
        "runs_completed": n_done,
        "runs_expected": total,
        "runs_this_session": n_new,
        "elapsed_seconds": round(elapsed, 2),
        "seconds_per_run": round(elapsed / n_new, 5) if n_new else None,
        "stopped_early": stopped_early,
        "reason": reason,
        "peak_rss_gb": round(rss_gb(), 2),
        "environment": {
            "chip": env_info.chip,
            "memory_gb": env_info.memory_gb,
            "os_version": env_info.os_version,
        },
        "trajectories": str(out_path),
    }
    (RESULTS / f"manifest_{stage}.json").write_text(json.dumps(manifest, indent=2))

    print("-" * 70)
    print(f"  runs completed    {n_done}/{total}"
          f"{'  (INCOMPLETE - ' + reason + ')' if stopped_early else ''}")
    print(f"  this session      {n_new}")
    print(f"  elapsed           {fmt_hms(elapsed)}")
    if n_new:
        per = elapsed / n_new
        print(f"  per run           {1000 * per:.1f} ms")
        print(f"  projected 5000 cases x {len(agents)} agents: "
              f"{fmt_hms(per * 5000 * len(agents))}")
    print(f"  peak RSS          {rss_gb():.2f} GB")
    print(f"  trajectories      {out_path}")
    print("=" * 70)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--n", type=int, default=None, help="override case count")
    ap.add_argument("--no-resume", action="store_true", help="discard prior checkpoint")
    args = ap.parse_args()

    # Fail loudly on numerical problems rather than propagating NaNs into
    # results tables. underflow is expected in log-space belief updates.
    np.seterr(all="raise", under="ignore")
    run_stage(args.stage, resume=not args.no_resume, override_n=args.n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Guards a results directory against mixing runs made under different settings.

Each results directory stores a manifest (_protocol.json) of the scoring, the
fold counts, the seed and any settings the caller adds. Writing into a
directory whose manifest does not match these stops the run.
"""

from __future__ import annotations

import json
import os

MANIFEST = "_protocol.json"


def _current(extra: dict | None = None) -> dict:
    from config import N_IN_CV, N_OUT_CV, RANDOM_STATE, SCORING

    # The inner-CV splitting policy differs between drivers, so the caller
    # states it through `extra`.
    out = {
        "scoring": SCORING,
        "n_out_cv": N_OUT_CV,
        "n_in_cv": N_IN_CV,
        "random_state": RANDOM_STATE,
    }
    if extra:
        out.update(extra)
    return out


def check(result_dir: str, extra: dict | None = None) -> dict:
    """Write the manifest, or refuse to write into a directory produced differently.

    `extra` carries whatever the caller knows that this module cannot infer --
    the inner-CV splitting policy, for one, which differs between the drivers.

    Returns the manifest. Raises SystemExit when an existing manifest disagrees,
    naming the fields that differ; the caller then either points --results-dir
    somewhere else or removes the stale results on purpose.
    """
    os.makedirs(result_dir, exist_ok=True)
    path = os.path.join(result_dir, MANIFEST)
    now = _current(extra)

    if os.path.exists(path):
        with open(path) as f:
            was = json.load(f)
        differing = {k: (was.get(k), now[k]) for k in now if was.get(k) != now[k]}
        if differing:
            lines = "\n".join(f"    {k}: stored {old!r}, now {new!r}"
                              for k, (old, new) in differing.items())
            raise SystemExit(
                f"{result_dir} holds results from a different protocol:\n{lines}\n"
                f"  Resuming would mix the two in the same files. Write to another\n"
                f"  --results-dir, or delete the stale results first."
            )
        return was

    # A directory that already has results but no manifest is refused.
    existing = [f for f in os.listdir(result_dir)
                if f.endswith(".json") and not f.startswith("_")]
    if existing:
        raise SystemExit(
            f"{result_dir} holds {len(existing)} result files but no {MANIFEST}, so\n"
            f"  the protocol behind them is unrecorded and cannot be compared with\n"
            f"  the current one. Write to another --results-dir, or record the\n"
            f"  protocol yourself if you know it matches."
        )

    with open(path, "w") as f:
        json.dump(now, f, indent=2)
    return now

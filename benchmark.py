"""Compare one baseline run with one optimized run in fresh processes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from run import video_files, write_json


def fixed_window(rows):
    """Measure twenty completions after twelve completed videos."""
    # Baseline groups contain four videos, so both boundaries fall between groups.
    ordered = sorted(rows, key=lambda row: row["completed_monotonic_s"])
    chosen = ordered[12:32]
    counts = Counter(row["video_id"] for row in chosen)
    if set(counts.values()) != {2}:
        raise ValueError("Unbalanced fixed20 window")
    seconds = ordered[31]["completed_monotonic_s"] - ordered[11]["completed_monotonic_s"]
    return dict(
        start_completed=12,
        end_completed=32,
        completed_videos=20,
        seconds=seconds,
        videos_per_second=20 / seconds,
        video_counts=dict(counts),
        job_ids=[row["job_id"] for row in chosen],
        generated_tokens=sum(row["generated_tokens"] for row in chosen),
    )


def compare(trials):
    baseline = trials["baseline"]["seconds20"]
    optimized = trials["optimized"]["seconds20"]
    return dict(
        variants=trials,
        throughput_increase_percent=(baseline / optimized - 1) * 100,
        time_reduction_percent=(1 - optimized / baseline) * 100,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--videos", nargs="+")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    try:
        files = video_files(source, args.videos)
        if len(files) != 10 or len(set(files)) != 10:
            raise ValueError("Expected ten different videos")
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise ValueError("Output directory must be absent or empty")
    except ValueError as error:
        parser.error(str(error))

    output.mkdir(parents=True, exist_ok=True)
    trials = {}
    for mode in ("baseline", "optimized"):
        trial = output / f"{mode}-r1"
        subprocess.run(
            [
                sys.executable, "-B", str(Path(__file__).with_name("run.py")),
                "--input", str(source), "--output", str(trial),
                "--mode", mode, "--benchmark", "--videos", *files,
            ],
            check=True,
        )
        report = json.loads((trial / "benchmark.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (trial / "stream.jsonl").read_text(encoding="utf-8").splitlines()]
        steady = fixed_window(rows)
        trials[mode] = dict(
            run=trial.name,
            seconds20=steady["seconds"],
            videos_per_second=steady["videos_per_second"],
            tokens20=steady["generated_tokens"],
            seconds50=report["seconds"],
            tokens50=report["generated_tokens"],
            truncated50=report["truncated"],
        )
    result = compare(trials)
    write_json(output / "comparison.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

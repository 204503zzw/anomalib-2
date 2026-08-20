# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the benchmark pipeline configuration handling."""

from pathlib import Path

import pandas as pd
import yaml
from jsonargparse import Namespace

from anomalib.pipelines.benchmark.job import BenchmarkJob
from anomalib.pipelines.benchmark.pipeline import Benchmark


class _FakeRunner:
    """Runner that records the arguments and output directory it was called with."""

    def __init__(self, calls: list[tuple[dict, Path | None]]) -> None:
        self.calls = calls
        self.generator = Namespace(job_class=BenchmarkJob)

    def run(self, args: dict, prev_stage_results: object = None) -> None:
        del prev_stage_results
        self.calls.append((args, BenchmarkJob._output_dir))  # noqa: SLF001


def _run_pipeline(tmp_path: Path, config: dict) -> list[tuple[dict, Path | None]]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    calls: list[tuple[dict, Path | None]] = []
    pipeline = Benchmark()
    pipeline._setup_runners = lambda _args: [_FakeRunner(calls)]  # type: ignore[method-assign]  # noqa: SLF001
    pipeline.run(Namespace(config=str(config_path)))
    return calls


def test_single_configuration_is_run_once(tmp_path: Path) -> None:
    """A config without ``runs`` is executed once with the default output directory."""
    calls = _run_pipeline(
        tmp_path,
        {"accelerator": "cpu", "benchmark": {"seed": 42, "model": {"class_path": "Padim"}}},
    )
    assert len(calls) == 1
    args, output_dir = calls[0]
    assert args["seed"] == 42
    assert output_dir is None


def test_runs_are_executed_sequentially_with_separate_output_dirs(tmp_path: Path) -> None:
    """Each entry of ``runs`` is executed in order and saved to its own directory."""
    calls = _run_pipeline(
        tmp_path,
        {
            "accelerator": "cpu",
            "benchmark": {
                "output_dir": str(tmp_path / "sweep"),
                "seed": 42,
                "model": {"class_path": "Padim", "init_args": {"backbone": "resnet18", "layers": ["layer1"]}},
                "runs": [
                    {"name": "first"},
                    {"name": "second", "seed": 7, "model": {"init_args": {"backbone": "resnet50"}}},
                ],
            },
        },
    )
    assert [output_dir for _, output_dir in calls] == [tmp_path / "sweep" / "first", tmp_path / "sweep" / "second"]

    first, second = (args for args, _ in calls)
    # Defaults are inherited, overrides are merged into the defaults.
    assert first["seed"] == 42
    assert first["model"]["init_args"] == {"backbone": "resnet18", "layers": ["layer1"]}
    assert second["seed"] == 7
    assert second["model"]["init_args"] == {"backbone": "resnet50", "layers": ["layer1"]}
    # Defaults are not mutated by a run's overrides.
    assert first["name"] == "first"


def test_save_writes_to_configured_output_directory(tmp_path: Path) -> None:
    """``output_directory`` redirects where ``results.csv`` is written."""
    result = pd.DataFrame({"image_AUROC": [1.0]})
    with BenchmarkJob.output_directory(tmp_path / "my_run"):
        BenchmarkJob.save(result)
    assert (tmp_path / "my_run" / "results.csv").is_file()
    assert BenchmarkJob._output_dir is None  # noqa: SLF001

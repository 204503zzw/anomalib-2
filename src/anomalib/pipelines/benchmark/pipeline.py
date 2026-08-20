# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmarking pipeline for evaluating anomaly detection models.

This module provides functionality for running benchmarking experiments that evaluate
and compare multiple anomaly detection models. The benchmarking pipeline supports
running experiments in parallel across multiple GPUs when available.

Configuration:
    The benchmark pipeline is configured via a YAML file passed through the --config
    command line argument. The config uses jsonargparse conventions with
    ``class_path`` and ``init_args`` for object instantiation.

    Example config structure::

        accelerator:
          - cpu
        benchmark:
          seed: 42
          model:
            grid:
              - class_path: Patchcore
          data:
            class_path: MVTecAD
            init_args:
              category:
                grid:
                  - bottle

    The ``grid`` key creates multiple jobs for each combination of values.

    Instead of a single (grid searched) configuration, a list of named runs can be
    given under ``runs``. The runs are executed one after the other and each one
    writes its own ``results.csv``::

        accelerator: cpu
        benchmark:
          output_dir: runs/my_experiment
          seed: 42
          data:
            class_path: MVTecAD
          runs:
            - name: padim
              model:
                class_path: Padim
            - name: patchcore
              model:
                class_path: Patchcore

    Keys defined next to ``runs`` (``seed`` and ``data`` above) are used as defaults
    for every run and can be overridden per run. Results are written to
    ``<output_dir>/<name>/results.csv``.

Example:
    Run the benchmark with a config file:

    Programmatically with explicit config path:

    >>> from benchmark.pipeline import Benchmark
    >>> from argparse import Namespace
    >>> args = Namespace(config="src/config.yaml")
    >>> results = Benchmark().run(args)

    Or via command line arguments (no args passed):

    >>> from benchmark.pipeline import Benchmark
    >>> # This will parse sys.argv looking for --config argument
    >>> results = Benchmark().run()

    Which expects the script to be called as:

    .. code-block:: bash

        python script.py --config src/config.yaml

The pipeline handles setting up appropriate runners based on available hardware,
using parallel execution when multiple GPUs are available and serial execution
otherwise.
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from jsonargparse import Namespace

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.base.pipeline import log_file
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner
from anomalib.utils.logging import redirect_logs

from .generator import BenchmarkJobGenerator
from .job import BenchmarkJob

if TYPE_CHECKING:
    from anomalib.pipelines.types import PREV_STAGE_RESULT

logger = logging.getLogger(__name__)


class Benchmark(Pipeline):
    """Benchmarking pipeline for evaluating anomaly detection models.

    This pipeline handles running benchmarking experiments that evaluate and compare
    multiple anomaly detection models. It supports both serial and parallel execution
    depending on available hardware.

    Example:
        >>> from anomalib.pipelines import Benchmark
        >>> from anomalib.data import MVTecAD
        >>> from anomalib.models import Padim, Patchcore

        >>> # Initialize benchmark with models and datasets
        >>> benchmark = Benchmark(
        ...     models=[Padim(), Patchcore()],
        ...     datasets=[MVTecAD(category="bottle"), MVTecAD(category="cable")]
        ... )

        >>> # Run benchmark
        >>> results = benchmark.run()
    """

    @staticmethod
    def _setup_runners(args: dict) -> list[Runner]:
        """Set up the appropriate runners for benchmark execution.

        This method configures either serial or parallel runners based on the
        specified accelerator(s) and available hardware. For CUDA devices, parallel
        execution is used when multiple GPUs are available.

        Args:
            args (dict): Dictionary containing configuration arguments. Must include
                an ``"accelerator"`` key specifying either a single accelerator or
                list of accelerators to use.

        Returns:
            list[Runner]: List of configured runner instances.

        Raises:
            ValueError: If an unsupported accelerator type is specified. Only
                ``"cpu"`` and ``"cuda"`` are supported.

        Example:
            >>> args = {"accelerator": "cuda"}
            >>> runners = Benchmark._setup_runners(args)
        """
        accelerators = args["accelerator"] if isinstance(args["accelerator"], list) else [args["accelerator"]]
        runners: list[Runner] = []
        for accelerator in accelerators:
            if accelerator not in {"cpu", "cuda"}:
                msg = f"Unsupported accelerator: {accelerator}"
                raise ValueError(msg)
            device_count = torch.cuda.device_count()
            if device_count <= 1 or accelerator == "cpu":
                runners.append(SerialRunner(BenchmarkJobGenerator(accelerator)))
            else:
                runners.append(ParallelRunner(BenchmarkJobGenerator(accelerator), n_jobs=device_count))
        return runners

    def run(self, args: Namespace | None = None) -> None:
        """Run the benchmark pipeline.

        When the configuration contains a ``runs`` list, each entry is executed
        sequentially and its results are saved to a separate directory. Otherwise the
        default single-configuration behaviour is used.

        Args:
            args (Namespace | None): Arguments to run the pipeline. These are the args
                returned by ``ArgumentParser``.
        """
        pipeline_args = self._get_args(args)
        benchmark_args = pipeline_args.get(BenchmarkJob.name) or {}
        runs = benchmark_args.get("runs")
        redirect_logs(log_file)

        if not runs:
            self._run_group(pipeline_args, benchmark_args, benchmark_args.get("output_dir"))
            return

        defaults = {key: value for key, value in benchmark_args.items() if key not in {"runs", "output_dir"}}
        output_dir = Path(benchmark_args.get("output_dir", Path("runs") / BenchmarkJob.name))
        for index, run in enumerate(runs):
            run_args = _merge(defaults, run)
            name = str(run_args.setdefault("name", f"run_{index}"))
            logger.info(f"Running benchmark run '{name}' ({index + 1}/{len(runs)})")
            self._run_group(pipeline_args, run_args, output_dir / name)

    def _run_group(self, pipeline_args: dict, job_args: dict, output_dir: str | Path | None) -> None:
        """Execute the runners once for a single set of job arguments."""
        previous_results: PREV_STAGE_RESULT = None
        for runner in self._setup_runners(pipeline_args):
            try:
                with BenchmarkJob.output_directory(output_dir):
                    previous_results = runner.run(job_args, previous_results)
            except Exception:  # noqa: PERF203 catch all exception and allow try-catch in loop
                logger.exception("An error occurred when running the runner.")
                print(
                    f"There were some errors when running {runner.generator.job_class.name} with"
                    f" {runner.__class__.__name__}."
                    f" Please check {log_file} for more details.",
                )


def _merge(defaults: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into a copy of ``defaults``."""
    merged = deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged

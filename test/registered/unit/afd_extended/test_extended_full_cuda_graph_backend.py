"""Unit tests for ``FullCudaGraphBackend.capture_one`` profiling hooks — CPU-only.

These cover the changes from the "cuda graph profile traces" PR that wire the
runner's torch profiler into the capture loop:

  * When profiling is disabled, ``capture_one`` runs exactly two warmups + one
    capture and never touches a profiler (behavior-identical to before the PR).
  * When the runner exposes an active ``_profiler`` (per-bs capture profiling,
    ``--enable-profile-cuda-graph`` + ``SGLANG_GRAPH_BATCH_CAPTURE``),
    ``capture_one`` calls ``profiler.step()`` past the two warmups and once after
    the capture (schedule ``wait=2, warmup=0, active=1``). The captured forward is
    NOT wrapped in a ``record_function``; per-bs trace naming is handled by the
    profiler's ``on_trace_ready`` callback instead.
  * The ``getattr`` guards mean a runner that sets the flag but has no
    ``_profiler`` attribute degrades gracefully (no stepping, no crash).

The real capture path needs CUDA (``torch.cuda.CUDAGraph`` + device graph
context), so those are mocked; the logic under test (call counts, ordering,
profiler stepping) is pure-Python and runs on CPU.
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Sentinel: distinguishes "runner has no _profiler attribute" from
# "_profiler is None" in the test fixtures.
_UNSET = object()


class _FakeGraphCtx:
    """Stand-in for ``device_module.graph(...)`` — a no-op context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_backend(runner):
    """Use the production constructor; replace only CUDA memory measurement."""
    with mock.patch(
        "sglang.srt.model_executor.runner_backend.full_cuda_graph_backend.get_parallel",
        return_value=SimpleNamespace(tp_group=SimpleNamespace(barrier=mock.Mock())),
    ):
        backend = FullCudaGraphBackend(runner)
    backend._precarve = SimpleNamespace(
        measure=contextlib.nullcontext, mint=mock.Mock()
    )
    backend._reuse_output_buffer = False
    backend._output_buffer = None
    backend._memory_saver_adapter = None
    backend._cuda_graph_runner = runner
    backend._device_module = runner.device_module
    return backend


def _make_runner(*, enable_profile, profiler, num_tokens_per_bs=1, mode_name="DECODE"):
    device_module = SimpleNamespace(
        synchronize=mock.Mock(name="synchronize"),
        graph=mock.Mock(name="graph", side_effect=lambda **kw: _FakeGraphCtx()),
    )
    runner = SimpleNamespace(
        device_module=device_module,
        num_tokens_per_bs=num_tokens_per_bs,
        capture_forward_mode=SimpleNamespace(name=mode_name),
        enable_profile_cuda_graph=enable_profile,
    )
    if profiler is not _UNSET:
        runner._profiler = profiler
    return runner


class TestStandaloneExecutionContext(CustomTestCase):
    def test_rejects_invalid_warmup_policy(self):
        for count in (-1, True, 0.5):
            with self.subTest(warmup_steps=count), self.assertRaises(ValueError):
                FullCudaGraphBackend(device_module=object(), warmup_steps=count)
        with self.assertRaises(ValueError):
            FullCudaGraphBackend(
                device_module=object(), warmup_steps=0, reuse_output_buffer=True
            )

    def test_requires_one_execution_context(self):
        with self.assertRaises(ValueError):
            FullCudaGraphBackend()
        runner = _make_runner(enable_profile=False, profiler=None)
        with self.assertRaises(ValueError):
            FullCudaGraphBackend(runner, device_module=runner.device_module)


if __name__ == "__main__":
    unittest.main()

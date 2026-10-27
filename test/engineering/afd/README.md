# Extended AFD tests

Additional CPU and serving regression tests for AFD. Shared test helpers live
under `sglang.test`.

## CPU regression tests

Run from the repository root in an environment with the project test
dependencies installed:

```bash
python -m pytest \
  test/registered/unit/afd \
  test/registered/unit/afd_extended \
  test/registered/unit/layer_boundary/test_afd_boundary.py \
  test/registered/unit/model_executor/model_runner_components/test_cuda_graph_setup.py \
  test/registered/unit/model_executor/runner_backend/test_full_cuda_graph_backend.py \
  test/registered/unit/model_executor/test_forward_batch_view.py \
  -q
```

The additional cases cover lifecycle and cleanup, asymmetric topology, routing,
metadata ownership, role graphs, model boundaries, and backend admission.
CPU cases substitute accelerator boundaries and do not validate CUDA kernels,
real NCCL communication, or model accuracy.

## GPU serving tests

The focused native/1A1F S2 test remains at
`test/registered/e2e/afd/test_qwen3_moe_serving.py`. The extended test adds eager
and graph configurations, output checks, graph-use records, and shutdown checks.
It requires three assigned compatible CUDA devices and cached model weights.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
export AFD_E2E_MODEL=/path/to/Qwen3-30B-A3B
export AFD_E2E_ATTENTION_BACKEND=fa4
export AFD_E2E_LOG_DIR=$(mktemp -d)
export AFD_E2E_RESULT_FILE="$AFD_E2E_LOG_DIR/results.json"
python test/registered/e2e/afd_extended/test_extended_qwen3_moe_serving.py
```

Choose an attention backend supported by the assigned hardware and installed
dependencies. These commands do not allocate devices. CPU checks and successful
test collection do not imply that the GPU tests have run.

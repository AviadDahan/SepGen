"""A DiffusionStage that yields an already-prepared transformer instead of building one.

WHY THIS EXISTS. Upstream's `DiffusionStage._transformer_ctx` returns
`gpu_model(self._build_transformer(...))`, and `gpu_model` calls `model.dispose()` on exit,
which replaces every parameter and persistent buffer with a **meta** tensor. That is correct
for a stage that rebuilds from a checkpoint each call, but our stage-1 transformer carries a
PEFT adapter plus five attention gates and costs minutes to assemble -- it must survive across
the cells of a campaign. Overriding this ONE private method keeps `DiffusionStage.__call__`
(the state building, the six-kwarg loop invocation, the post-loop clear_conditioning +
unpatchify) byte-for-byte stock, which is the entire point of this package.

Residency is the caller's job: move the prepared model to CPU before any VAE work and back
before the next cell. `_build_transformer` raises so that an upstream refactor which stops
routing through `_transformer_ctx` fails loudly instead of silently assembling a second 22B.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from ltx_core.model.transformer.model import X0Model
from ltx_pipelines.utils.blocks import DiffusionStage

# Fail at import if the upstream seam we override is renamed.
if not hasattr(DiffusionStage, "_transformer_ctx"):  # pragma: no cover - guard
    raise RuntimeError(
        "ltx_pipelines DiffusionStage no longer defines _transformer_ctx; "
        "stempipe/prepared_stage.py must be re-pointed at the new seam")


class PreparedDiffusionStage(DiffusionStage):
    """`DiffusionStage` whose transformer is supplied, cached, and never disposed."""

    def __init__(self, prepared: X0Model, builder, dtype: torch.dtype, device: torch.device,
                 **kwargs) -> None:
        # The real builder is still handed to the base class: __call__ logs
        # `self._transformer_builder.checkpoint`, and `_assert_supports_conditionings`
        # consults `model_config()` on every call.
        super().__init__(builder, dtype, device, **kwargs)
        self._prepared = prepared

    @property
    def prepared(self) -> X0Model:
        return self._prepared

    def to_gpu(self) -> None:
        self._prepared.to(self._device)

    def to_cpu(self) -> None:
        self._prepared.to("cpu")
        torch.cuda.empty_cache()

    def assert_live(self) -> None:
        """Catch a stray dispose() before it produces a cell of garbage."""
        for name, p in self._prepared.named_parameters():
            if p.is_meta:
                raise RuntimeError(
                    f"prepared transformer parameter {name} is on META -- something disposed "
                    "the cached model; every subsequent render would be invalid")
            break  # one probe is enough: dispose() metas the whole module at once

    def _transformer_ctx(self, **kwargs) -> Iterator[X0Model]:  # noqa: D401 - upstream seam
        self.assert_live()
        self._prepared.to(self._device)

        @contextmanager
        def _ctx():
            yield self._prepared          # deliberately NO dispose / no offload here

        return _ctx()

    def _build_transformer(self, **kwargs):  # pragma: no cover - guard
        raise RuntimeError(
            "PreparedDiffusionStage never builds a transformer; it was constructed with one. "
            "If upstream started calling _build_transformer directly, re-point this class.")

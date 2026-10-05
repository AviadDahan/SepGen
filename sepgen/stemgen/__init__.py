"""stemgen -- the SepGen joint-stem training strategy, sampler and attention gates for LTX-2.5.

Adds two stem spans to the audio sequence of the LTX-2.5 audio-video generator, next to the
audio-mix span. Built on the LTX trainer's training-strategy extension point, so LoRA setup,
quantization, gradient checkpointing, checkpointing and resume come from upstream unchanged.
"""

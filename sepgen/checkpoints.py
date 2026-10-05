"""SepGen checkpoint names -> local files, downloaded from Hugging Face on first use.

    sep-12k  AviadDahan/SepGen-Separation   separation checkpoint (12,000 steps, clean audio-mix)
    gen-3k   AviadDahan/SepGen-Generation   generation checkpoint (sep-12k + 3,000 steps, noisy audio-mix)

Each repo holds lora_weights.safetensors and a config.json describing the adapter; config.json is
fetched with the weights (it is also what the Hub counts as a download).
"""
from pathlib import Path

CHECKPOINTS = {
    "sep-12k": "AviadDahan/SepGen-Separation",
    "gen-3k": "AviadDahan/SepGen-Generation",
}
WEIGHTS_FILE = "lora_weights.safetensors"


def resolve_checkpoint(name_or_path: str) -> Path:
    """`sep-12k` / `gen-3k` -> the downloaded LoRA file; anything else must be a local path."""
    if name_or_path in CHECKPOINTS:
        from huggingface_hub import hf_hub_download
        repo = CHECKPOINTS[name_or_path]
        hf_hub_download(repo, "config.json")
        return Path(hf_hub_download(repo, WEIGHTS_FILE))
    path = Path(name_or_path)
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path} (or use one of {sorted(CHECKPOINTS)})")
    return path

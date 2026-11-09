"""3DINO attention LoRA: study-based adapters and five-class MLP head."""

try:
    from .cli import main
except ImportError:  # direct invocation
    from cli import main


if __name__ == "__main__":
    raise SystemExit(main("lora"))

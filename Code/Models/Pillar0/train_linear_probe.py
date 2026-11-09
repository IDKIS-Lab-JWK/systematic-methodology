"""Pillar-0 frozen-encoder linear-probe entry point."""

try:
    from .cli import main
except ImportError:  # pragma: no cover
    from cli import main


if __name__ == "__main__":
    raise SystemExit(main("linear_probe"))

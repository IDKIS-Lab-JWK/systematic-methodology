"""COLIPRI frozen-encoder pooling-optimised probe entry point."""

try:
    from .cli import main
except ImportError:  # pragma: no cover
    from cli import main


if __name__ == "__main__":
    raise SystemExit(main())

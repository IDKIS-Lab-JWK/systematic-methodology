#!/usr/bin/env python
"""Run Merlin full fine-tuning with grid, five-fold CV, and held-out testing."""

from cli import main


if __name__ == "__main__":
    main("full_finetune")

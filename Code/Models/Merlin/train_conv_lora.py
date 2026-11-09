#!/usr/bin/env python
"""Run Merlin ConvLoRA fine-tuning with grid, five-fold CV, and held-out testing."""

from cli import main


if __name__ == "__main__":
    main("conv_lora")

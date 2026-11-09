"""CLI for FlexiCT final-quarter fine-tuning with independent result identity."""

import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from FlexiCT.final_quarter_finetune import FlexiCTFinalQuarterAdapter
from FlexiCT.train_final_block_finetune import _add_arguments, _validate
from _supervised_adaptation.cli import main_for


def main(argv=None):
    return main_for(
        argv, adapter_factory=FlexiCTFinalQuarterAdapter,
        add_model_arguments=_add_arguments, validate_model_arguments=_validate,
    )


if __name__ == "__main__":
    raise SystemExit(main())

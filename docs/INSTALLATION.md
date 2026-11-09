# Installation

Run commands from the repository root. CLI help and data/configuration planning
use lightweight dependencies; numerical training and prediction use the model's
isolated environment. Do not merge the seven model requirements into one venv.

| Model | Historical environment target | Requirements |
|---|---|---|
| 3DINO | Windows x64, Python 3.10, PyTorch 2.0/CUDA 11.7 | `Code/Models/ThreeDINO/requirements.txt` |
| FlexiCT | Python 3.11, PyTorch 2.8/CUDA 12.8 | `Code/Models/FlexiCT/requirements.txt` |
| COLIPRI | Python 3.11, model-specific pinned packages | `Code/Models/COLIPRI/requirements.txt` |
| Pillar0 | model-specific transformer/RAVE environment | `Code/Models/Pillar0/requirements.txt` |
| Merlin | merlin-vlm 0.0.7 and its compatible PyTorch stack | `Code/Models/Merlin/requirements.txt` |
| MedicalNet / DenseNet | shared CNN environment | `Code/Models/_cnn/requirements.txt` |

Use an interpreter compatible with your selected model, for example:

```sh
python scripts/setup_environment.py --model FlexiCT
```

The helper creates `.venv_flexict` and installs its requirements. It performs
package downloads only when explicitly run. Install the appropriate GPU-enabled
PyTorch wheel for your machine where the model requirements do not pin an index.
Use the resulting environment's Python to run the wrapper.

Gated model access requires your own authorization and local credentials.
Pretrained backbones are not bundled. Supply required local backbone/upstream
paths through the runner's asset options. Existing pinned setup/provisioning
modules document source revisions and verify checkpoint identities. Do not
substitute a different backbone for a head-only or partial checkpoint.

The historical 3DINO stack is Windows-specific. Linux support for that exact
stack is not claimed. GPU memory needs vary substantially by backbone, tuning
scope and input geometry; a CPU smoke test does not establish GPU readiness.

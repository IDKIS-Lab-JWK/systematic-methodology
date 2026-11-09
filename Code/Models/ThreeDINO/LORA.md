# 3DINO LoRA for five-class CT classification

Implemented 16 October 2026. Entry point: [train_lora.py](train_lora.py).
Adapters and head: [lora.py](lora.py). Exact defaults and attribution:
[default_lora.json](default_lora.json).

The recipe follows the released classification code for
[Do Medical Foundation Models Generalize on the African Brain?](https://papers.miccai.org/miccai-2026-sat/AFRICAI_011.html),
available as an AFRICAI/MICCAI 2026 camera-ready paper.
[Inspected source](https://anonymous.4open.science/r/MFM-African-Brain-78C8/src/lora/lora.py):
16 October 2026; displayed source commit date 29 June 2026, file version marker
`ac496fc9` (not a Git commit SHA). These are the released defaults; they are not
independently verified command-line settings for every published result.

## Training scope and settings

| Setting | Default |
|---|---|
| Backbone | Existing strictly loaded, pinned 3DINO ViT-L checkpoint |
| Adapter targets | `attn.qkv` and `attn.proj` in all 24 blocks; 48 wrapped linear modules |
| LoRA | Rank 4, alpha 1, scale alpha/r = 0.25, no adapter dropout |
| Initialization | Kaiming-uniform A and zero B; initially zero adapter contribution |
| Frozen parameters | All original encoder weights, biases, norms, embeddings and tokens |
| Representation | Final normalized CLS token, dimension 1024 |
| Classifier | 1024 -> 128 -> 64 -> 5; BatchNorm, ReLU, dropout 0.3 after each hidden layer |
| Optimizer | Adam, LR 0.0001 for adapters and head, betas (0.9, 0.999), epsilon 1e-8, weight decay 0 |
| Budget | 50 epochs per development fold; final-budget checkpoint, no early stopping |
| Scheduler | Constant LR; no warm-up or decay |
| Batch | 4 per process, accumulation 1, shuffled finite epochs, drop last training batch |
| Precision | FP16 CUDA autocast with gradient scaling; FP32 on CPU |
| Loss | Unweighted five-class cross-entropy |
| Seed / workers | 42 / 0 |

The code uses an MLP classifier although the paper describes a linear classifier.
This implementation follows the **released MLP**, changing its output to five
classes. It does not reuse the existing linear-probe head. The class order is
Appendicitis, Cholezystitis, Normal, Perforation, Sigmoid.

The adapter count calculated for the verified 24-block, width-1024 architecture
is 589,824 parameters. The five-class MLP has 140,165 trainable parameters:
729,989 trainable parameters total, excluding non-trainable BatchNorm buffers.

Our existing CT preprocessing is retained: RAS orientation, percentile scaling,
deterministic 144 x 144 x 112 prefix cache, random augmented 112-cubed training
crops and centered evaluation crops. This differs from the study's normalized
96-cubed MRI inputs. Cached encoder features cannot be used to train LoRA.

## Commands

Run from `Systematic_Methodology`. These commands are examples, not evidence of
completed training. Choose the desired existing percentage manifest explicitly.

Windows, plan only (does not load weights or write results):

```powershell
.\.venv_3dino\Scripts\python.exe Code\Models\ThreeDINO\train_lora.py --stage cv --percentage 100 --dry_run
```

Windows, five development folds and development refit:

```powershell
.\.venv_3dino\Scripts\python.exe Code\Models\ThreeDINO\train_lora.py --stage cv --percentage 100 --device cuda --resume
```

Linux/Slurm, inside an allocated single-GPU job, using an installed compatible
Python/PyTorch environment and explicitly supplied cluster paths:

```bash
python Code/Models/ThreeDINO/train_lora.py \
  --stage cv --percentage 100 --device cuda --resume --local_files_only \
  --dataset_root /workspace/project/Dataset \
  --results_root /workspace/project/Results/3DINO \
  --cache_root /workspace/project/Results/3DINO/Cache \
  --upstream_root /workspace/project/upstream/3DINO \
  --pretrained_checkpoint /workspace/project/weights/3dino_vit_weights.pth
```

The paths above are placeholders. The cache root must contain the existing
sealed `Shared/<namespace>` deterministic-prefix cache, and the dataset root
must contain the expected manifests and patient layout. The local Windows
environment itself is not portable to Linux. H200 execution and the Linux
dependency stack require validation on the cluster; this implementation does
not establish their readiness. The runner is single-process/single-GPU; it does
not automatically distribute across eight GPUs. Do not launch multiple jobs
against the same CV registry/output directory concurrently.

For held-out evaluation after development training, use the same paths/profile
with `--stage test --model_source refit_full_development --resume`.
`cv_ensemble` and `best_cv_fold` also use the existing evaluation workflow.
`--stage all` performs development training followed by held-out evaluation.

For explicit changes, copy the default JSON to a separate file, keep
`"method": "lora"`, and supply `--parameter_source manual --parameter_config PATH`.
There is no publication/grid mode for LoRA because the source provides a fixed
recipe rather than a search grid. The BatchNorm MLP requires an actual training
micro-batch of at least two; gradient accumulation cannot make batch size one
equivalent. Scheduler/optimizer changes, adapter dropout and gradient
checkpointing are rejected rather than silently ignored.

## Results, evaluation and resumption

Results use `Results/3DINO/LoRA/CV/CV_x/<percentage>_percent/` and
`Results/3DINO/LoRA/Test/CV_x/<model_source>/<percentage>_percent/`.
Existing frozen linear-probe results remain separate.

The existing patient-isolated development folds, CV registry, completion seals,
refit and held-out report formats are reused. The source's five random 50:50
splits are not adopted. Validation diagnostics and checkpoints are recorded
each epoch, but LoRA selects the **final training budget**, not the best epoch.
The shared filename `best_model.pt` and legacy `best_optimizer_step` metadata
therefore refer to the final-budget checkpoint for LoRA. Development refit uses
the existing median fold update budget, not a new search on held-out data.

Checkpoints contain adapters, all classifier parameters and BatchNorm buffers,
optimizer/scaler/constant-LR-counter state, RNG and committed data cursor.
Frozen backbone weights are reconstructed from the bound original checkpoint;
they are not duplicated in adapter checkpoints. Target-module and state-key
checks reject incompatible checkpoints. The runtime/parameter fingerprints
reject incompatible resume settings, and non-resume direct training refuses to
overwrite an existing run.

Synthetic tests cover adapter gradients, unchanged original weights, exact
adapter/head round trips including BatchNorm state, interrupted-run resumption,
prediction, refit, CLI planning and invalid settings. They do not establish
full-size GPU memory fit, clinical performance or a completed 3DINO experiment.

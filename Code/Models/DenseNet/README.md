> Historical study guide. Commands below document the original protocol.
> Cached legacy test/all dispatch is disabled in this distribution. Use
> `python run.py test` from the repository root for uncached source-image
> inference. Legacy training remains available with an explicit CV/grid stage.
> The generic new-data training recipe is documented in `docs/USAGE.md`.

# DenseNet121: 3D training from scratch

This is MONAI 1.5.1 `DenseNet121(spatial_dims=3, in_channels=1, out_channels=5)`:
growth rate 32, initial features 64, blocks 6/12/24/16, random initialization.
It has 11,247,749 parameters. Both this runner and the older `densenet/`
implementation are **3D**; Huang et al.'s original paper is **2D**.

The [shared CNN methodology and run guide](../../../CNN_BASELINES.md) documents
the current patient distribution, train-only spacing fit, coverage/memory
decision, all preprocessing rationale, outputs, timing and run commands.
The shared input policy gives this baseline the same geometric/intensity
treatment as MedicalNet while keeping fold statistics independent.

`publication.json` uses the original paper's ImageNet optimizer family,
Nesterov momentum 0.9, decay 0.0001, 90 epochs and tenfold LR drops at 30/60.
LR 0.01, batch 4, volumetric input, HU processing and weighted loss are task
adaptations. Early termination is disabled so both milestones are reached;
the best validation checkpoint is retained. Neither the paper nor MONAI
establishes the best 3D dimensions for this task.

`default.json` offers AdamW 0.0001, dropout 0.2 and patience 10 as a separately
labeled practical profile. `grid.json` varies LR 0.003/0.01/0.03;
`resolution_grid.json` compares 192³ and 256³ on development folds only.

```powershell
# From Systematic_Methodology:
.\CNN_Run_Script.ps1 -Model DenseNet -Percentages 20 -Stage all
```

Default results: `Results/DenseNet/FromScratch`. The old outside-folder results
are historical single-split experiments with different patient distributions;
they must not be presented as results from this systematic five-fold protocol.

> Historical study guide. Commands below document the original protocol.
> Cached legacy test/all dispatch is disabled in this distribution. Use
> `python run.py test` from the repository root for uncached source-image
> inference. Legacy training remains available with an explicit CV/grid stage.
> The generic new-data training recipe is documented in `docs/USAGE.md`.

# MedicalNet: full 3D fine-tuning

Use the [shared CNN methodology and run guide](../../../CNN_BASELINES.md) for
commands, exact dataset distribution, every preprocessing rationale, storage,
resume and output layout. The implementation uses the existing systematic
five-fold CV/refit/test workflow, not the old physical percentage folders.

The encoder is the official dilated ResNet18 with shortcut A, at upstream commit
`20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd`. `vendor/resnet.py` and the MIT license
are preserved; [vendor/provenance.json](vendor/provenance.json) records hashes.
The existing `resnet_18_23dataset.pth` must have SHA-256
`61224f9317fcce873366deb3703183e92cc47325b726b69691b33536244e10f4`.
The **23-dataset weights are an official code release**; the Med3D paper
describes 8-dataset pretraining. Loading requires every one of the 102 encoder
tensors with the expected shape. Only a segmentation decoder, if present,
may be omitted. The global-average/5-class linear head is initialized anew.

The official spatial strides and dilations are retained. The local shortcut-A
wrapper replaces upstream's detached `out.data`/`Variable` path with a
dtype-preserving differentiable concatenation. Tests check identical forward
values against the vendored implementation and nonzero shortcut gradients.
This is explicitly an autograd fix, not a change to released weight shapes.

`publication.json` anchors transfer Adam/LR 0.001 in Med3D §4.2. Its 200-epoch
cap comes from official demo code; early stopping, effective batch 4, 256³
canvas, class weighting and [0,1] intensity mapping are task adaptations.
`default.json`, `grid.json` and `resolution_grid.json` document separate
practical and development-search options. Their provenance fields are inert.

```powershell
# From Systematic_Methodology:
.\CNN_Run_Script.ps1 -Model MedicalNet -Percentages 20 -Stage all
```

Default results: `Results/MedicalNet/FullFineTune`. Actual spacing is fitted
from each fold's training patients; validation/test use it unchanged, expanding
the padded canvas when needed. All rationales and applied dimensions are saved.

# Licensing and third-party notices

The project's original software is licensed under the [MIT License](LICENSE).
Existing third-party notices and licences remain in force. The root MIT grant
must not be read as relicensing external model weights, clinical images,
radiologist review text or other third-party materials.

`Code/Models/MedicalNet/vendor/` retains the adapted MedicalNet source's original
MIT licence and provenance, including the local shortcut-autograd correction.

The model-specific profiles, provenance and source record upstream repositories,
pinned revisions, papers and checkpoint identities. Important source locations:

| Family | Upstream source / model |
|---|---|
| Merlin | https://github.com/StanfordMIMI/Merlin |
| Pillar0 | https://huggingface.co/YalaLab/Pillar0-AbdomenCT |
| 3DINO | https://github.com/AICONSlab/3DINO and https://huggingface.co/AICONSlab/3DINO-ViT |
| FlexiCT | https://github.com/ricklisz/FlexiCT |
| COLIPRI | https://huggingface.co/microsoft/colipri |
| MedicalNet | https://github.com/Tencent/MedicalNet |
| DenseNet / MONAI | https://github.com/Project-MONAI/MONAI |

Pretrained backbones are not bundled. Frozen-head/adaptation exports still need
their compatible original backbone, including any upstream access approval.
Public project-checkpoint availability does not grant access to a gated
upstream backbone or change its terms. Checkpoints retain their original model
provenance; the software MIT licence is not an independent weights licence.

The experiment branch's case-study images and clinical interpretations are
provided as the study's publication material, using public R01-R20 labels.
They are outside the software-only MIT grant. No separate blanket reuse licence
for those clinical materials is asserted here. Original clinical datasets,
identifier linkage and verbatim correspondence are not distributed.

# Performed study protocol

The study-specific protocol is distinct from the generic framework. `study.json`
records class order, analysis configuration, cohort sizes and original split
hashes. `../configs/study/` records the resolved parameters for every reported
model/method/development-size combination; no patient assignment is included.

All models share the same fixed 278-patient held-out test set. The primary report
uses five-fold probability ensembles. Best-validation-fold and development-refit
routes are sensitivity analyses. Perforation remains at 40 development patients
at every size, so these sizes change both sample count and class composition.

`Data_Splitting_Plan.txt` retains the original planning document. Its earlier
70/30 rows are historical planning values; the performed protocol uses five
mutually exclusive CV folds. Use its final CV section and `study.json` as the
execution description. Real split membership remains private.

The retained `Code/Dataset/generate_split_manifests.py` is the historical,
cohort-specific generator. Never substitute newly generated assignments when
claiming an exact reproduction of the original splits: compare against the
recorded original hashes and use the authorized original manifests.

Parameter provenance distinguishes publication-derived, publication-adapted and
project choices. The retrospective Merlin FullFineTune CV_2 addition and the
corrected Merlin ConvLoRA_LLRD CV_2 remain separate from historical alternatives.
Current framework code is not assumed to be the code used by every historical
run. Checkpoints retain recorded original source identities separately from
new distributable-file hashes.

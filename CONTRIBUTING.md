# Contributing

Use the public runner and an isolated environment for the model you change.
Keep synthetic tests separate from private input data. Include a focused test
for changes to split safety, checkpoint interpretation, preprocessing, metrics,
or training scope. Record method changes explicitly; a generic new training
recipe must not be represented as a reproduction of the published experiment.

Do not add clinical source data, caches, credentials, local audit outputs,
verbatim correspondence, editor state or assistant working files. New datasets,
training products and notebooks stay local unless explicitly prepared for a
publication release. Report test commands and their actual results in changes.

# Epilepsy

This folder holds our experiment where the goal is to extract known Disease-Gene relationships in respect to Epilepsy. The entuire task has been divided into three parts specified below.

## The Scripts
1. Data Processing [[Data Parser File](epilepsy-data-parser.ipynb)]: This file does the heavy lifting in regards to tagging/parsing the given corpora. Also, this extracts candidates from the tagged text as well.

2. Label Functions [[Labeler File](epilepsy-labeler.ipynb)]: This file contains our label functions that act as a weak supervisor for predicting candidate relationships.

3. Classification [[ML Run File](epilepsy-ml-run.ipynb)]: This file runs the machine learning algorithms for predicting candidates.

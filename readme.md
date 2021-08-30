# Orthography Standardization Models

This repository contains all of the models trained in the scope of my master's thesis which revolves around the standardization of orthography for Arabic dialects. This [pdf document](https://drive.google.com/file/d/1q3So1h3cUpOC76m6h4GWLpTrgxK9YyiP/view?usp=sharing) contains Chapters 2 (Preliminary Modeling) and 5 (Joint Learning Using Multiple Features) of my thesis report in which all of the models and results are described in detail. Aleternatively, this [link](https://drive.google.com/file/d/1VA4PZ1UKKQmpJXi0JYkh8miuOsNMDk-U/view?usp=sharing) takes you to the full thesis report.

## Disclaimer
This repository has yet to be polished and is still in the same state in which it was left at submission time. It was not in any way made to be user-friendly, and was pushed for the time being for purely administrative reasons. It will soon be overhauled and released in a form which is more readily usable.

## Rough Guide
Below are some instructions on how to find the code for all the models, in the order in which they are found in Chapters 2 and 5 of the thesis report. Each of the following directories contains the code for the training, evaluation, and modeling scripts for the respective models.

### Merger Model
1. `git checkout master`
2. Found in the `alignment_handler` directory

### Word-level BERT2BERT Model
1. `git checkout taxonomy`
2. Found in the `bert2bert` directory

### Character-level and Hybrid Standardization Models
1. `git checkout taxonomy`
2. The following models can be found:
    - Hybrid BERT2BERT with characer decoder in the `bert2hybert` directory
    - Character-level and hybrid character-level with BERT word-level context in the `spell_correct` directory

### Segmenter Model
1. `git checkout segmenter`
2. Found in the `segmentation` directory

### Modular Multi-task Model
1. `git checkout taxonomy`
2. Found in the `spell_correct` directory
3. Training script is `train_pos.py`
4. Different ablations can be ran by changing the `args.mode` and `args.features` variables:

| Ablation  | `args.mode` | `args.features` |
| --------- | ----------- | --------------- |
| POS Tagger | `'tagger'` | `'pos'` |
| Morphological Tagger | `'tagger'` | `'pos person gender number ...'` |
| Orthography Standardizer | `'standardizer'` | NA |
| Joint Standardizer and Morphological Tagger | `'joint'` | `'pos person gender ...'` |
| Taxonomy Tagger | 'taxonomy' | NA |

## Introduction

This repository contains Python code and pretrained models for Vehicle-pedestrian-interaction-based iterative prediction framework for pedestrian's trajectory presented in our paper.

## Requirements

tensorflow (tested with 1.9 and 1.14)

keras (tested with 2.1 and 2.2)

scikit-learn

numpy

pillow

## Training

```sh
python train_test.py --data /path/to/data
```

`--data [dir]`, data directory

## Test

```sh
python train_test.py --data /path/to/data --test
```

`--data [dir]`, data directory

`--test`, test flag




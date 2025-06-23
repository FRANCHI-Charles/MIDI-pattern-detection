# MIDI-pattern-detection

This repository contain the code of the work on
**Pattern Extraction on Discrete Spaces: Combining Machine Learning and Mathematical Morphology for Computational Music Analysis**.
You can read the report as `Report.pdf`.

## How to use this repository

Install the python library in `requirements.txt` using pip, and run the following files to see the different experiments:
- `checkfugues.ipynb` is the basic data analysis of the Fugues dataset.
- `bachtest.ipynb` is the experiment of Section 6 to find pattern variations.
- `simplepatternlearnertest.ipynb` is the experiment ran to tune the Correlation loss function (see Section 7.2 of the report).
- `patternlearner.ipynb` is the CNN training and visualisation of the model results (see Section 7.3 of the report).

## Organisation of the repository

- `CNN_model` cotains the model files of the CNN as Pytorch neural networks. They can be load with `torch.load`.
  - `which_parameters.md` details the parameters used for each model during traning.
- `dataset` contains function to process MIDI files.
- `Fugues_data` contains:
  - the Fugue dataset as pickle format (`data_{8,16,48}[_reduced].pkl`), with `{8,16,48}` the minimum beat division, "reduced" version contain 10 less files (see the report). When opened, the data object is a dictionary with key the name of the music `BWV_***` or `op87_**b`, and as value a list of (onset, pitch) of notes.
  - `midi_to_pkl.py` processes the raw data to create other `.pkl` version of the dataset.
  - Other files are utils library and raw data.
- `ML` contains the Pattern learner architecture (experiment in Section 7 of the report).
- `variations` contains functions for the experiment in Section 6 of the report.
- `bachtest.ipynb` is the experiment of Section 6 to find pattern variations.
- `checkfugues.ipynb` is the basic data analysis of the Fugues dataset.
- `cnn_training.py` is the main file to run to train a new CNN. You need to change the path of the dataset so you can use it.
- `patternlearner.ipynb` is the jupyter notebook version of `cnn_training.py` and contain visualisation of the model results.
- `simplepatternlearnertest.ipynb` is the experiment ran to tune the Correlation loss function (see Section 7.2 of the report).
- `requirements.txt` is the library needed to run the code.

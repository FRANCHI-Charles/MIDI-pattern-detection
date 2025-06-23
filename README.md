# MIDI-pattern-detection

This repository contains the code supporting the work:  
**Pattern Extraction on Discrete Spaces: Combining Machine Learning and Mathematical Morphology for Computational Music Analysis**.  
You can find the accompanying report in `Report.pdf`.

## How to Use This Repository

Install the Python libraries listed in `requirements.txt` using pip. Then, run the following notebooks to explore the different experiments:

- `checkfugues.ipynb` — Basic data analysis of the Fugues dataset.  
- `bachtest.ipynb` — Experiment from Section 6 for pattern variations.  
- `simplepatternlearnertest.ipynb` — Tuning the correlation loss function (see Section 7.2 of the report).  
- `patternlearner.ipynb` — CNN training and model result visualization (see Section 7.3 of the report).

## Repository Structure

- `CNN_model/`  
  Contains the CNN model files implemented with PyTorch. Models can be loaded with `torch.load`.  
  - `which_parameters.md` documents the training parameters used for each model.

- `dataset/`  
  Contains functions for processing MIDI files.

- `Fugues_data/`  
  - Fugue datasets in pickle format: `data_{8,16,48}[_reduced].pkl`, where `{8,16,48}` refers to the minimum beat division. The "reduced" versions contain 10 fewer files (see the report for details).  
    Each file is a dictionary where keys are music piece names (e.g., `BWV_***`, `op87_**b`) and values are lists of note (onset, pitch) pairs.  
  - `midi_to_pkl.py` converts raw data into `.pkl` format.  
  - Other files include utility functions and raw MIDI data.

- `ML/`  
  Contains the architecture of the Pattern Learner (related to Section 7 of the report).

- `variations/`  
  Includes functions for the experiment in Section 6.

- `bachtest.ipynb` — See above.  
- `checkfugues.ipynb` — See above.  
- `cnn_training.py` — Main script for training a new CNN. You will need to update the dataset path accordingly.  
- `patternlearner.ipynb` — Jupyter notebook version of `cnn_training.py` with visualizations.  
- `simplepatternlearnertest.ipynb` — See above.  
- `requirements.txt` — List of required Python libraries.

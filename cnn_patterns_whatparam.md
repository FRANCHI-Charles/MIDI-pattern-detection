# Parameters of each model

## cnn_patterns_1

```python
SEED = 689

DATA_PATH = "data_8_reduced.pkl"
MINDIV = 8 #from Fugues_data.midi_to_pkl import MINDIV

CONV_BIASES = True
BATCHNORM_AFFINE = True
DENSE_BIAS = True

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
CNN_MODEL_NAME = 'cnn_patterns_'
PATTERNS_MAXSIZE = (1, 4*8, 13)

MAX_EPOCH = 10000
BATCH_SIZE = 15
LEARNING_RATE = 0.01
PATIENCE = 5
REFINEMENT = 3  # restart training after patience runs out with the best model, decrease lr by...
LR_FAC = 0.1    # ... the learning rate factor lr_fac: lr_new = lr_old*lr_fac
LOG_INTERVAL = 60  # seconds

OPTIMIZER = Adam
LOSS_FUNCTION = CorrelationLoss().square_regul(beta=0.8, smooth_function=3, mean_size=8)
```

## cnn_patterns_2

```python
SEED = 689

DATA_PATH = "data_8_reduced.pkl"
MINDIV = 8 #from Fugues_data.midi_to_pkl import MINDIV

CONV_BIASES = True
BATCHNORM_AFFINE = True
DENSE_BIAS = True

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
CNN_MODEL_NAME = 'cnn_patterns_'
PATTERNS_MAXSIZE = (1, 4*8, 13)

MAX_EPOCH = 10000
BATCH_SIZE = 1
LEARNING_RATE = 0.01
PATIENCE = 5
REFINEMENT = 3  # restart training after patience runs out with the best model, decrease lr by...
LR_FAC = 0.1    # ... the learning rate factor lr_fac: lr_new = lr_old*lr_fac
LOG_INTERVAL = 60  # seconds

OPTIMIZER = Adam
LOSS_FUNCTION = CorrelationLoss().square_regul(beta=0.8, smooth_function=3, mean_size=8)
```


## cnn_patterns_3

```python
SEED = 689

DATA_PATH = "data_8_reduced.pkl"
MINDIV = 8 #from Fugues_data.midi_to_pkl import MINDIV

CONV_BIASES = False
BATCHNORM_AFFINE = True
DENSE_BIAS = False #not enough alone

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
CNN_MODEL_NAME = 'cnn_patterns_'
PATTERNS_MAXSIZE = (1, 4*8, 13)

MAX_EPOCH = 10000
BATCH_SIZE = 1
LEARNING_RATE = 0.01
PATIENCE = 5
REFINEMENT = 3  # restart training after patience runs out with the best model, decrease lr by...
LR_FAC = 0.1    # ... the learning rate factor lr_fac: lr_new = lr_old*lr_fac
LOG_INTERVAL = 60  # seconds

OPTIMIZER = Adam
LOSS_FUNCTION = CorrelationLoss().square_regul(beta=0.8, smooth_function=3, mean_size=8)
```


## cnn_patterns_4

```python
SEED = 689

DATA_PATH = "data_8_reduced.pkl"
MINDIV = 8 #from Fugues_data.midi_to_pkl import MINDIV

CONV_BIASES = False
BATCHNORM_AFFINE = False
DENSE_BIAS = False #not enough alone

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
CNN_MODEL_NAME = 'cnn_patterns_'
PATTERNS_MAXSIZE = (1, 4*8, 13)

MAX_EPOCH = 10000
BATCH_SIZE = 1
LEARNING_RATE = 0.01
PATIENCE = 5
REFINEMENT = 3  # restart training after patience runs out with the best model, decrease lr by...
LR_FAC = 0.1    # ... the learning rate factor lr_fac: lr_new = lr_old*lr_fac
LOG_INTERVAL = 60  # seconds

OPTIMIZER = Adam
LOSS_FUNCTION = CorrelationLoss().square_regul(beta=0.8, smooth_function=3, mean_size=8)
```


## cnn_patterns_4bis

```python
SEED = 689

DATA_PATH = "data_8_reduced.pkl"
MINDIV = 8 #from Fugues_data.midi_to_pkl import MINDIV

CONV_BIASES = False
BATCHNORM_AFFINE = False
DENSE_BIAS = False #not enough alone

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
CNN_MODEL_NAME = 'cnn_patterns_'
PATTERNS_MAXSIZE = (1, 4*8, 13)

MAX_EPOCH = 10000
BATCH_SIZE = 1
LEARNING_RATE = 0.01
PATIENCE = 5
REFINEMENT = 3  # restart training after patience runs out with the best model, decrease lr by...
LR_FAC = 0.1    # ... the learning rate factor lr_fac: lr_new = lr_old*lr_fac
LOG_INTERVAL = 60  # seconds

OPTIMIZER = Adam
LOSS_FUNCTION = CorrelationLoss().square_regul(beta=0.8, smooth_function=3, mean_size=8)
```

## cnn_patterns_5

```python
SEED = 689

DATA_PATH = "data_8_reduced.pkl"
MINDIV = 8 #from Fugues_data.midi_to_pkl import MINDIV

CONV_BIASES = False
BATCHNORM_AFFINE = False
DENSE_BIAS = False #not enough alone

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
CNN_MODEL_NAME = 'cnn_patterns_'
PATTERNS_MAXSIZE = (1, 4*8, 13)

MAX_EPOCH = 10000
BATCH_SIZE = 1
LEARNING_RATE = 0.01
PATIENCE = 5
REFINEMENT = 3  # restart training after patience runs out with the best model, decrease lr by...
LR_FAC = 0.1    # ... the learning rate factor lr_fac: lr_new = lr_old*lr_fac
LOG_INTERVAL = 60  # seconds

OPTIMIZER = Adam
LOSS_FUNCTION = CorrelationLoss().square_regul(beta=0, smooth_function=3, mean_size=8)
```
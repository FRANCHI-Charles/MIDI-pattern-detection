# All-in-one file to train cnn on cluster.
# Please be aware Local function may not be up to date.

import os
from time import time
import pickle

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.functional as TF
import torch.nn as nn
from torch.nn.functional import sigmoid, relu, conv2d, conv3d
import math


################ GLOBAL VARIABLES ####################

SEED = 689

DATA_PATH = "data_16_reduced.pkl"
MINDIV = 16 #from Fugues_data.midi_to_pkl import MINDIV

TEST_SIZE = 0.1
VALIDATION_SIZE = 0.2
CNN_MODEL_NAME = 'cnn_patterns_'
PATTERNS_MAXSIZE = (1, 4*8, 13)

MAX_EPOCH = 10000
BATCH_SIZE = 15
LEARNING_RATE = 0.1
PATIENCE = 5
REFINEMENT = 3  # restart training after patience runs out with the best model, decrease lr by...
LR_FAC = 0.1    # ... the learning rate factor lr_fac: lr_new = lr_old*lr_fac
LOG_INTERVAL = 60  # seconds




torch.manual_seed(SEED)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    #raise EnvironmentError("cuda is not available.")


## Local function from other librairies

# from ML.utils
def _erodila_conv(ar, selem, device, convdim):
    while ar.ndim < 2 + convdim:
        ar = ar.unsqueeze(0)
    while selem.ndim < 2 + convdim:
        selem = selem.unsqueeze(0)
    ar = ar.float().to(device)
    selem = selem.float().to(device)
    out = list()
    if convdim == 2:
        for i in range(ar.shape[0]):
            out.append(conv2d(ar[i].unsqueeze(0), selem[i].unsqueeze(0), padding=(selem.shape[-2] // 2, selem.shape[-1] // 2)).squeeze(0))
        return torch.stack(out)
    else:
        for i in range(ar.shape[0]):
            out.append(conv3d(ar[i].unsqueeze(0), selem[i].unsqueeze(0), padding=(selem.shape[-3] // 2, selem.shape[-2] // 2, selem.shape[-1] // 2)).squeeze(0))
        return torch.stack(out)


def _selem_sum(selem, convdim):
    if convdim == 3:
        selem_sum = selem.sum((-3,-2,-1))
        while selem_sum.ndim < 5:
            selem_sum = selem_sum.unsqueeze(-1)
    else:
        selem_sum = selem.sum((-2,-1))
        while selem_sum.ndim < 4:
            selem_sum = selem_sum.unsqueeze(-1)
    return selem_sum
    

def correlation(ar: np.ndarray | torch.Tensor, selem: np.ndarray | torch.Tensor, device: torch.device = "cpu", convdim : int = 2, return_numpy_array: bool = False):
    """
    Perform a correlation using torch.conv[2,3]d.

    Parameters
    ----------
    ar : np.ndarray | torch.Tensor
        Image to erode, with shape ((T,) H, W ), (Channel, (T,) H, W) or (MiniBatch, Channel, (T,) H, W) (T if `convdim = 3`).
    selem : np.ndarray | torch.Tensor
        Element S to erode `ar` with, with shape ((T,) H,W), (Channel, (T,) H, W) or (MiniBatch, Channel, (T,) H, W) (T if `convdim = 3`).
    device : torch.device
        Device to send the `ar` and `selem` tensors.
    convdim : int = 2, in [2,3]
        Dimension of the convolution to perform.
    return_numpy_array : bool = False
        Convert the output in numpy.ndarray.

    Outputs
    -------
    torch.Tensor | np.ndarray
        Tensor of dimension (MiniBatch=1, Channel=1, (T,) H, W).
    """
    # torch_array = (_old_erodila_conv(ar, selem, device) == selem.sum()).squeeze((0,1))
    if not isinstance(ar, torch.Tensor):
        ar = torch.tensor(ar)
    if not isinstance(selem, torch.Tensor):
        selem = torch.tensor(selem)

    conv_results = _erodila_conv(ar, selem, device, convdim)
    if selem.shape[-1] %2 == 0:
        conv_results = conv_results[..., 1:]
    if selem.shape[-2] %2 == 0:
        conv_results = conv_results[..., 1:, :]
    if convdim == 3 and selem.shape[-3] %2 == 0:
        conv_results = conv_results[..., 1:, :, :]

    selem_sum = _selem_sum(selem, convdim)
    torch_array = conv_results / selem_sum

    if return_numpy_array:
        return torch_array.to("cpu").int().numpy()
    return torch_array


#from dataset.utils
def load_pickle_data(filename:str):
    with open(filename, "rb") as file :
        return pickle.load(file)


#from Fugues_data.loader
class FuguesDataset(Dataset): 
    """
    Data Augmentation with flip pitches.
    NOTE : implement pitch transposition and time translation when possible.
    
    """

    def __init__(self, data_file:str, maxpitchdif:int=59, maxlength:float=481.0, mindiv:int=16):
        unprocessed = load_pickle_data(data_file)
        self.mindiv = mindiv
        self.names = list()
        self.data = list()

        for name, value in unprocessed.items():
            minpitch = min([note[1] for note in value])
            mintime = min([note[0] for note in value])
            translated = [(note[0] - mintime, note[1] - minpitch) for note in value]
            track = [note for note in translated if note[0] <= maxlength and note[1] <= maxpitchdif]

            matrix = torch.zeros((1, int((maxlength+1)*self.mindiv), maxpitchdif+1), dtype=torch.int8)
            for point in track:
                matrix[0, round((point[0])*self.mindiv), point[1]] = 1
            
            self.names.append(name)
            self.data.append(matrix)

        self.data = torch.stack(self.data)

    def __getitem__(self, index):
        return torch.concat((self.data, torch.flip(self.data, (-1,))))[index]
    
    def __len__(self):
        return self.data.shape[0] * 2
    

# from ML.architecture
class CorrelationLoss():
    """
    Class of Differents correlation loss, function that tries to maximize the translative correlation between the output (kernel) and the input (image).
    
    - absolute_regul
    - square_regul
    - minmax_regul
    """

    def __init__(self, **kwargs):
        self.beta = kwargs.get("beta", 0.)
        self.smooth_function = kwargs.get("smooth_function", None)
        self.gamma = kwargs.get("gamma", 0.)
        self.mean_size = kwargs.get("mean_size", None)


    def minmax_regul(self, smooth_function:float=None, beta:float=None, gamma:float=None):
        """
        The loss is `-(1000 * correlation_sum/input.sum() + beta * output.sum()/output_shape + gamma * difference_between_patterns/pattern_size)` and need to be minimize.

        Parameters
        ----------
        smooth_function : float|Callable
            The function to regularize importance of correlation values.
            If float or int (None => 3), use f(tensor) = tensor ** smooth_function.
            Else, the function must take a tensor in input and return a tensor of same dimension.
        beta:float
            Regularization parameters to minimize (positive value)/maximize (negative value) the sizeof the pattern in the output.
        gamma:float
            Regularization parameters such as all patterns per channel are differents.

        Return
        ------
        Callable
            The loss function that takes in parameters (output, image).
    
        """
        if smooth_function is not None:
            self.smooth_function = smooth_function
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma

        def loss(output:torch.Tensor, input:torch.Tensor):
            """
            output : the output patterns of dimension (OutChannels, Channel//groups, H, W)
            
            input : the original image to correlate with.
            """
            added_cor = self._added_correlation(output, input)

            difference_term, normalisation = self._gamma_regul(output)

            return -(1000*added_cor/input.sum() - self.beta * output.sum()/math.prod(output.shape) + self.gamma * difference_term /normalisation)
            
        return loss
    

    def square_regul(self, smooth_function:float=None, beta:float=None, gamma:float=None, mean_size:float = None):
        """
        The loss is `-(1000 * correlation_sum/input.sum() + size_regulation + gamma * difference_between_patterns/pattern_size)` and need to be minimize.
        Wtih size_regulation is the MSE between the total size and the mean_size if float.

        Parameters
        ----------
        smooth_function : float|Callable
            The function to regularize importance of correlation values.
            If float or int (None => 3), use f(tensor) = tensor ** smooth_function.
            Else, the function must take a tensor in input and return a tensor of same dimension.
        beta:float
            Regularization apply to the MSE of the difference between the size (sum) of the output and the expected one.
        gamma:float
            Regularization parameters such as all patterns per channel are differents.
        mean_size:float
            The mean size of the patterns to be learned.

        Return
        ------
        Callable
            The loss function that takes in parameters (output, image).
        """
        if smooth_function is not None:
            self.smooth_function = smooth_function
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if mean_size is not None:
            self.mean_size = mean_size

        def loss(output:torch.Tensor, input:torch.Tensor):
            """
            output : the output patterns of dimension (OutChannels, Channel//groups, H, W).
            
            input : the original image to correlate with.
            """
            added_cor = self._added_correlation(output, input)

            difference_term, normalisation = self._gamma_regul(output)

            size_loss = torch.tensor(0.)
            for i in range(output.shape[0]):
                size_loss += ((output[i,0][output[i,0] >= 0.5]).sum() - self.mean_size)**2
            size_loss = size_loss/output.shape[0]
            return -(1000*added_cor/input.sum() - self.beta/1000 * size_loss + self.gamma * difference_term /normalisation)
        
        return loss
    

    def absolute_regul(self, smooth_function:float=None, beta:float=None, gamma:float=None, mean_size:float = None):
        """
        The loss is `-(1000 * correlation_sum/input.sum() + size_regulation + gamma * difference_between_patterns/pattern_size)` and need to be minimize.
        Wtih size_regulation is the absolute difference between the total size and the mean_size if float.

        Parameters
        ----------
        smooth_function : float|Callable
            The function to regularize importance of correlation values.
            If float or int (None => 3), use f(tensor) = tensor ** smooth_function.
            Else, the function must take a tensor in input and return a tensor of same dimension.
        beta:float
            Regularization apply to the absolute difference between the size (sum) of the output and the expected one.
        gamma:float
            Regularization parameters such as all patterns per channel are differents.
        mean_size:float
            The mean size of the patterns to be learned.

        Return
        ------
        Callable
            The loss function that takes in parameters (output, image).
        
        """
        if smooth_function is not None:
            self.smooth_function = smooth_function
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if mean_size is not None:
            self.mean_size = mean_size

        def loss(output:torch.Tensor, input:torch.Tensor):
            """
            output : the output patterns of dimension (OutChannels, Channel//groups, H, W).
            
            input : the original image to correlate with.
            """
            added_cor = self._added_correlation(output, input)

            difference_term, normalisation = self._gamma_regul(output)

            size_loss = torch.tensor(0.)
            for i in range(output.shape[0]):
                size_loss += torch.abs((output[i,0][output[i,0] >= 0.5]).sum() - self.mean_size)
            size_loss = size_loss/output.shape[0]
            return -(1000*added_cor/input.sum() - self.beta/100 * size_loss + self.gamma * difference_term /normalisation)
        
        return loss


    def _added_correlation(self, output, input):
        if len(output.shape) !=4:
            raise ValueError("output must be of dimension (OutChannels, Channel//groups, H, W)")
        cor = correlation(input, output)
        added_cor = self.smooth_function(cor).sum()
        return added_cor
    

    def _gamma_regul(self, output):
        difference_term = torch.tensor(0.)
        normalisation = torch.tensor(1.)
        if self.gamma != 0:
            normalisation = torch.tensor(math.prod(output[:,0].shape) * len(output.shape[1])*(len(output.shape[1])-1)/2)
            for i in range(output.shape[1]):
                for j in range(i+1, output.shape[1]):
                    difference_term += torch.abs(output[:,i] - output[:,j]).sum()

        return difference_term, normalisation
    

    @property
    def smooth_function(self):
        return self._smooth_function
    
    @smooth_function.setter
    def smooth_function(self, value):
        if value is None:
            self._smooth_function = lambda tensor : tensor ** 3
        elif isinstance(value, int) or isinstance(value, float):
            self._smooth_function = lambda tensor : tensor ** value
        else:
            self._smooth_function = value


class PatternLearner(nn.Module):
    """DO WE USE RESIDUAL ?
    
    Dropout if needed."""


    def __init__(self, input_shape:tuple[int,int,int], output_shape:tuple[int,int,int], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", torch.float32)
        factory_kwargs = {'device': self.device, 'dtype': self.dtype}

        self.conv_size = (9,13) # time : dividor of mindiv + 1 - pitches : 1 octava
        self.conv_padding = (self.conv_size[0]//2, self.conv_size[1]//2)
        self.maxpool_size = (4,1) # Compress time, not pitches
        self.maxpool_lastsize = (4,4)
        self.maxpool_dilatation = (1,13) # Dilatation on octava for last pooling
        self.nbr_channels = 3

        self.conv1 = nn.Conv2d(1, self.nbr_channels, self.conv_size, padding=self.conv_padding)
        self.maxpool1 = nn.MaxPool2d(self.maxpool_size)
        # ReLU
        self.batchnorm1 = nn.BatchNorm2d(self.nbr_channels)

        self.conv2 = nn.Conv2d(self.nbr_channels, 2*self.nbr_channels, self.conv_size, padding=self.conv_padding)
        self.maxpool2 = nn.MaxPool2d(self.maxpool_size)
        # ReLU
        self.batchnorm2 = nn.BatchNorm2d(2*self.nbr_channels)

        self.conv3 = nn.Conv2d(2* self.nbr_channels, 4* self.nbr_channels, self.conv_size, padding=self.conv_padding)
        self.maxpool3 = nn.MaxPool2d(self.maxpool_size)
        # ReLU
        self.batchnorm3 = nn.BatchNorm2d(4*self.nbr_channels)

        self.conv4 = nn.Conv2d(4 * self.nbr_channels, 8 * self.nbr_channels, self.conv_size, padding=self.conv_padding)
        self.maxpool4 = nn.MaxPool2d(self.maxpool_lastsize, dilation=self.maxpool_dilatation)
        # ReLU
        self.batchnorm4 = nn.BatchNorm2d(8*self.nbr_channels)

        # view
        self._features_in = self._get_features_size()
        self._features_in = math.prod(self._features_in[-3:])
        self.dense5 = nn.Linear(self._features_in, math.prod(output_shape))
        # view
        self.boost = nn.Parameter(torch.empty((1,), **factory_kwargs))
        # Sigmoid


    def _get_features_size(self):
        with torch.no_grad():
            input = torch.rand(1, *self.input_shape)
            output = self._forward_conv(input)
            return output.shape
        

    def _forward_conv(self, x, debug=False):
        output = self.conv1(x)
        output = self.maxpool1(output)
        output = relu(output)
        output = self.batchnorm1(output)
        if debug:
            print(f"Shape after conv 1 : {output.shape}")

        output = self.conv2(output)
        output = self.maxpool2(output)
        output = relu(output)
        output = self.batchnorm2(output)
        if debug:
            print(f"Shape after conv 2 : {output.shape}")

        output = self.conv3(output)
        output = self.maxpool3(output)
        output = relu(output)
        output = self.batchnorm3(output)
        if debug:
            print(f"Shape after conv 3 : {output.shape}")

        output = self.conv4(output)
        output = self.maxpool4(output)
        output = relu(output)
        output = self.batchnorm4(output)
        if debug:
            print(f"Shape after conv 4 : {output.shape}")

        return output
    

    def forward(self, x, debug=False):
        output = self._forward_conv(x, debug)

        output = output.view(-1, self._features_in)
        output = self.dense5(output)
        output = sigmoid(self.boost * output)
        if debug:
            print(f"Shape after dense Layer : {output.shape}")
        
        output = output.view(-1, *self.output_shape)

        return output


####### PATTERN LEARNER DIRECT FUNCTION #######

def train_epoch_cnn(model, optimizer):
    """
    Training loop for one epoch of NN training.
    """
    model.train()  # set model to training mode (activate dropout layers if any)
    t = time() # we measure the needed time
    for batch_idx, input_data in enumerate(train_loader):  # iterate over training input_data
        input_data = input_data.float().to(device)  # move input_data to device (GPU) if necessary
        optimizer.zero_grad()  # reset optimizer
        output = model(input_data)   # forward pass: calculate output of network for input_data
        loss = LOSS_FUNCTION(output, input_data)

        loss.backward()  # backward pass: calculate gradients using automatic diff. and backprop.
        optimizer.step()  # udpate parameters of network using our optimizer
        cur_time = time()
        # print some outputs if we reached our logging interval
        if cur_time - t > LOG_INTERVAL or batch_idx == len(train_loader)-1:  
            print(f"[{batch_idx * BATCH_SIZE}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]",
                  f"\tloss: {loss.item():.6f}, took {cur_time - t:.2f}s")
            t = cur_time


def valid_cnn(model):
    """
    Test loss evaluation
    """
    model.eval()  # set model to inference mode (deactivate dropout layers)
    with torch.no_grad():  # do not calculate gradients since we do not want to do updates
        output = model(test_data)
        loss = LOSS_FUNCTION(output, test_data)
    print(f'Average eval loss: {loss:.4f}\n')
    return loss


def train_cnn():
    """
    Run CNN training using the datasets.

    Return
    ------
        nn.Model
            trained model
    """

    # create model and optimizer, we use plain SGD with momentum
    optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)

    model_cnt = 0
    new_model_file = os.path.join(CNN_MODEL_NAME + str(model_cnt) + '.model')
    while os.path.exists(new_model_file):
        model_cnt += 1
        new_model_file = os.path.join(CNN_MODEL_NAME + str(model_cnt) + '.model')
    

    # train model for max_epochs epochs, output loss after log_intervall seconds.
    # for each epoch run once on validation set, 
    # write model to disk if validation loss decreased
    # if validation loss increased, check for early stopping with patience and refinements
    # after model is trained, perform a run on test set and output loss (don't forget to reload best model!)
    best_valid_loss = 9999.
    cur_patience = PATIENCE
    cur_refin = REFINEMENT

    #model.load_state_dict(torch.load(last_model_file, map_location=device).state_dict())
    print('Training CNN...')
    start_t = time()

    for epoch in range(1, MAX_EPOCH+1):
        train_epoch_cnn(model, optimizer)
        valid_loss = valid_cnn(model)

        if valid_loss < best_valid_loss:
            torch.save(model, new_model_file)
            best_valid_loss = valid_loss
            cur_patience = PATIENCE

        elif cur_patience <=0:
            model.load_state_dict(torch.load(new_model_file, map_location=device).state_dict())
            if cur_refin <= 0:
                print("Max refinement reached !")
                break
            else:
                print("Max patience reached !")
                
                cur_patience = PATIENCE
                for param_group in optimizer.param_groups:
                    lr = LR_FAC * param_group['lr']
                    param_group['lr'] = lr
                cur_refin -= 1
            
        else:
            print("We still have patience...")
            cur_patience -= 1
    
    print(f'Training took: {time()-start_t:.2f}s for {epoch} epochs')
    
    return model



def load_cnn(load_model:str):
    "Load the model."
    if load_model is None or not os.path.exists(load_model):
        print('Model file not found, unable to load...')
    else:
        model.load_state_dict(torch.load(load_model, map_location=device).state_dict())
        print("Model file loaded: {}".format(load_model))
    return model


##################### CODE ####################

### DATA LOADING

data = FuguesDataset(DATA_PATH, mindiv=MINDIV)

kwargs = {'num_workers': 0} #{'num_workers': 4, 'pin_memory': True}
train, validation, test_data = random_split(data, [(1-TEST_SIZE) * (1-VALIDATION_SIZE), (1-TEST_SIZE) * VALIDATION_SIZE, TEST_SIZE])
valid_data = next(iter(DataLoader(validation, len(validation), **kwargs))).float().to(device)
train_loader = DataLoader(train, BATCH_SIZE, shuffle=True, **kwargs)
test_data = next(iter(DataLoader(test_data, len(test_data), **kwargs))).float().to(device)

### ARCHITECTURE LOADING

model = PatternLearner(data[0].shape, PATTERNS_MAXSIZE)

OPTIMIZER = Adam
LOSS_FUNCTION = CorrelationLoss().square_regul(beta=0.8, smooth_function=3, mean_size=8)


train_cnn()
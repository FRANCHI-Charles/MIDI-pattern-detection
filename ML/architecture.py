from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, relu
import math
from tqdm import tqdm

from ML.utils import correlation


### Loss functions
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


    def minmax_regul(self, smooth_function:float|Callable=None, beta:float=None, gamma:float=None):
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
            output : the output patterns of dimension (Batch, OutChannels, H, W)
            
            input : the original image to correlate with.
            """
            added_cor = self._added_correlation(output, input)

            difference_term, normalisation = self._gamma_regul(output)

            return -(1000*added_cor/input.sum() - self.beta * output.sum()/math.prod(output.shape) + self.gamma * difference_term /normalisation)
            
        return loss
    

    def square_regul(self, smooth_function:float|Callable=None, beta:float=None, gamma:float=None, mean_size:float = None):
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
            output : the output patterns of dimension (Batch, OutChannels, H, W).
            
            input : the original image to correlate with.
            """
            added_cor = self._added_correlation(output, input)

            difference_term, normalisation = self._gamma_regul(output)

            size_loss = 0.
            for i in range(output.shape[0]):
                size_loss += ((output[i,0][output[i,0] >= 0.5]).sum() - self.mean_size)**2
            size_loss = size_loss/output.shape[0]
            return -(1000*added_cor/input.sum() - self.beta/1000 * size_loss + self.gamma * difference_term /normalisation)
        
        return loss
    

    def absolute_regul(self, smooth_function:float|Callable=None, beta:float=None, gamma:float=None, mean_size:float = None):
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
            output : the output patterns of dimension (Batch, OutChannels, H, W).
            
            input : the original image to correlate with.
            """
            added_cor = self._added_correlation(output, input)

            difference_term, normalisation = self._gamma_regul(output)

            size_loss = 0.
            for i in range(output.shape[0]):
                size_loss += torch.abs((output[i,0][output[i,0] >= 0.5]).sum() - self.mean_size)
            size_loss = size_loss/output.shape[0]
            return -(1000*added_cor/input.sum() - self.beta/100 * size_loss + self.gamma * difference_term /normalisation)
        
        return loss


    def _added_correlation(self, output, input):
        if len(output.shape) !=4:
            raise ValueError("output must be of dimension (Batch, OutChannels, H, W)")
        cor = correlation(input, output)
        added_cor = self.smooth_function(cor).sum()
        return added_cor
    

    def _gamma_regul(self, output):
        difference_term = 0.
        normalisation = 1.
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


### ML Architecture
class SimplePatternLearner(nn.Module):

    def __init__(self, pattern_maxsize:tuple, strong_negative=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", torch.float32)
        factory_kwargs = {'device': self.device, 'dtype': self.dtype}

        while len(pattern_maxsize) < 4:
            pattern_maxsize = (1, *pattern_maxsize)

        self._pattern = nn.Parameter(torch.empty(pattern_maxsize, **factory_kwargs))
        self._boost = nn.Parameter(torch.empty((1,), **factory_kwargs))
        self.reset()

        # if strong_negative:
        #     if isinstance(strong_negative, int):
        #         value = strong_negative
        #     else:
        #         value = 10
        #     self.activation = lambda x: sigmoid(x[x<0])
        # else:
        self.activation = sigmoid


    def reset(self):
        nn.init.kaiming_uniform_(self._pattern, a=math.sqrt(5)) #same init as convolution
        nn.init.uniform_(self._boost, -1, 1)


    def correlation(self, image):
        with torch.no_grad():
            return correlation(image, self.pattern)
    

    def learn_pattern(self, image, loss:Callable=None, learning_rates:float=0.01, optimization:torch.optim.Optimizer=torch.optim.Adam, maxepoch:int = 10**3, epsilon:float=0.001):
        self.image = image.to(self.device)
        
        self.optimizer = optimization(self.parameters(), lr=learning_rates)
        if loss is None:
            self.loss_func = CorrelationLoss().minmax_regul()
        else:
            self.loss_func = loss

        self.losses_list = [200,100]
        
        for epoch in tqdm(range(1,maxepoch+1), desc=f"Training...", ncols=100):

            if epsilon is not None and abs(self.losses_list[-2] - self.losses_list[-1]) <= epsilon:
                print(f"Epsilon was reached in {epoch-1} fits.")
                break

            self.losses_list.append(self.loss_func(self.activation(self._boost * self._pattern), image))

            self.losses_list[-1].backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch +=1

            if epoch%(maxepoch//10) ==0:
                print(f"Current_loss = {self.losses_list[-1]}")

        with torch.no_grad():
            print(f"Last loss : {self.losses_list[-1]:.8f}")

        self.losses_list = self.losses_list[2:]

    @property
    def pattern(self):
        with torch.no_grad():
            return self.activation(self._boost * self._pattern)

    

class PatternLearner(nn.Module):
    """DO WE USE RESIDUAL ?
    
    Dropout if needed."""


    def __init__(self, input_shape:tuple[int,int,int],
                 output_shape:tuple[int,int,int],
                 conv_size=None,
                 nbr_channels=None,
                 maxpool_size=None,
                 maxpool_lastsize=None,
                 maxpool_dilatation=None,
                 biases_conv = True,
                 learnable_batch_norm = True,
                 bias_dense = True,
                 *args, **kwargs):
        
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.device = kwargs.get("device", torch.device("cpu"))
        self.dtype = kwargs.get("dtype", torch.float32)
        factory_kwargs = {'device': self.device, 'dtype': self.dtype}

        self.conv_size = conv_size if conv_size is not None else (9,13) # time : dividor of mindiv + 1 - pitches : 1 octava
        self.conv_padding = (self.conv_size[0]//2, self.conv_size[1]//2)
        self.maxpool_size = maxpool_size if maxpool_size is not None else (6,1) # Compress time, not pitches
        self.maxpool_lastsize = maxpool_lastsize if maxpool_lastsize is not None else (6,4)
        self.maxpool_dilatation = maxpool_dilatation if maxpool_dilatation is not None else (1,13) # Dilatation on octava for last pooling
        self.nbr_channels = nbr_channels if nbr_channels is not None else 2

        self.conv1 = nn.Conv2d(1, self.nbr_channels, self.conv_size, padding=self.conv_padding, bias=biases_conv)
        self.maxpool1 = nn.MaxPool2d(self.maxpool_size)
        # ReLU
        self.batchnorm1 = nn.BatchNorm2d(self.nbr_channels, affine=learnable_batch_norm)

        self.conv2 = nn.Conv2d(self.nbr_channels, 2*self.nbr_channels, self.conv_size, padding=self.conv_padding, bias=biases_conv)
        self.maxpool2 = nn.MaxPool2d(self.maxpool_size)
        # ReLU
        self.batchnorm2 = nn.BatchNorm2d(2*self.nbr_channels, affine=learnable_batch_norm)

        self.conv3 = nn.Conv2d(2* self.nbr_channels, 4* self.nbr_channels, self.conv_size, padding=self.conv_padding, bias=biases_conv)
        self.maxpool3 = nn.MaxPool2d(self.maxpool_size)
        # ReLU
        self.batchnorm3 = nn.BatchNorm2d(4*self.nbr_channels, affine=learnable_batch_norm)

        self.conv4 = nn.Conv2d(4 * self.nbr_channels, 8 * self.nbr_channels, self.conv_size, padding=self.conv_padding, bias=biases_conv)
        self.maxpool4 = nn.MaxPool2d(self.maxpool_lastsize, dilation=self.maxpool_dilatation)
        # ReLU
        self.batchnorm4 = nn.BatchNorm2d(8*self.nbr_channels, affine=learnable_batch_norm)

        # view
        self._features_in = self._get_features_size()
        self._features_in = math.prod(self._features_in[-3:])
        self.dense5 = nn.Linear(self._features_in, math.prod(output_shape), bias=bias_dense)
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
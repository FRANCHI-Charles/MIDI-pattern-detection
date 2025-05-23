from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn.functional import sigmoid, relu
import math
from tqdm import tqdm

from ML.utils import correlation


def Correlation_loss(smooth_function:float|Callable=None, beta:float=1, gamma:float=0, mean_size:float = None):
    """
    Define a loss function that tries to maximize the translative correlation between the output (kernel) and the input (image).
    The loss is `-(1000 * correlation_sum/input.sum() + size_regulation + gamma * difference_between_patterns/pattern_size)` and need to be minimize.
    Wtih size_regulation is `beta * output.sum()/output_size` if mean_size = None, or the MSE between the total size and the mean_size if float.

    Parameters
    ----------
    smooth_function : float|Callable
        The function to regularize importance of correlation values.
        If float or int (None => 3), use f(tensor) = tensor ** smooth_function.
        Else, the function must take a tensor in input and return a tensor of same dimension.
    beta:float
        Regularization parameters to minimize the size of the pattern in the output.
        If `mean_size` if a float, regularization apply to the MSE of the difference between the size and the expected one.
    gamma:float
        Regularization parameters such as all patterns per channel are differents.
    mean_size:float
        The mean size of the patterns to be learned.
        If None, the size of the pattern is minimize.
        Else, the loss is the MSE between the total size and the mean_size.

    Return
    ------
    Callable
        The loss function that takes in parameters (output, image).
    
    """
    if smooth_function is None:
        smooth_function = lambda tensor : tensor ** 3
    if isinstance(smooth_function, int) or isinstance(smooth_function, float):
        exponent = smooth_function
        smooth_function = lambda tensor : tensor ** exponent

    if mean_size is None:
        def loss(output:torch.Tensor, input:torch.Tensor):
            """
            output : the output patterns
            
            input : the original image to correlate with.
            """
            cor = correlation(input, output)
            added_cor = smooth_function(cor).sum()

            difference_term = torch.tensor(0.)
            normalisation = torch.tensor(1.)
            if gamma != 0:
                normalisation = torch.tensor(math.prod(output[:,0].shape) * len(output.shape[1])*(len(output.shape[1])-1)/2)
                for i in range(output.shape[1]):
                    for j in range(i+1, output.shape[1]):
                        difference_term += torch.abs(output[:,i] - output[:,j]).sum()

            return -(1000*added_cor/input.sum() - beta * output.sum()/math.prod(output.shape) + gamma * difference_term /normalisation)
    else:
        def loss(output:torch.Tensor, input:torch.Tensor):
            """
            output : the output patterns of dimension (Batch, Channel, H, W)
            
            input : the original image to correlate with.
            """
            if len(output.shape) !=4:
                raise ValueError("output must be of dimension (Batch, Channel, H, W)")
            cor = correlation(input, output)
            added_cor = smooth_function(cor).sum()

            difference_term = torch.tensor(0.)
            normalisation = torch.tensor(1.)
            if gamma != 0:
                normalisation = torch.tensor(math.prod(output[:,0].shape) * len(output.shape[1])*(len(output.shape[1])-1)/2)
                for i in range(output.shape[1]):
                    for j in range(i+1, output.shape[1]):
                        difference_term += torch.abs(output[:,i] - output[:,j]).sum()

            size_loss = torch.tensor(0.)
            for i in range(output.shape[1]):
                size_loss += ((output[:,i].sum((-2,-1)) - mean_size)**2).sum()
            size_loss = size_loss/(output.shape[0]*output.shape[1])
            return -(1000*added_cor/input.sum() - beta/1000 * size_loss + gamma * difference_term /normalisation)
    
    return loss



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
            return correlation(image, self.pattern, self.device)
    

    def learn_pattern(self, image, loss:Callable=None, learning_rates:float=0.01, optimization:torch.optim.Optimizer=torch.optim.Adam, maxepoch:int = 10**3, epsilon:float=0.001):
        self.image = image
        
        self.optimizer = optimization(self.parameters(), lr=learning_rates)
        if loss is None:
            self.loss_func = Correlation_loss()
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

            if epoch%10 ==0:
                print(f"Current_loss = {self.losses_list[-1]}")

        with torch.no_grad():
            print(f"Last loss : {self.losses_list[-1]:.8f}")

        self.losses_list = self.losses_list[2:]

    @property
    def pattern(self):
        with torch.no_grad():
            return self.activation(self._boost * self._pattern)

    

        
        
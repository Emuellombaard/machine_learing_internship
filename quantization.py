import math
from functools import partial

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

#turns an int into a power of two
def unpack(exponent):
    if (exponent == 0):
        return 0.0
    absolute = 0 - abs(exponent)
    value = math.copysign(2 ** absolute, exponent)
    return value

#saves the model in int8
def save_quantized_model(model, filepath):
    parameters = []
    for name, layer in model.named_parameters():
        parameters.append(layer.clone().type(torch.int8)) 
    torch.save(parameters, filepath)

#loads an int8 model
def load_quantized_model(model, filepath):
    parameters = torch.load(filepath)
    unpacked_parameters = []
    for item in parameters:
        device = item.data.device
        unpacked = item.clone().type(torch.float32).cpu().apply_(unpack).to(device)
        unpacked_parameters.append(unpacked)

    i = 0
    for name, layer in model.named_parameters():
        layer.data = unpacked_parameters[i].clone()
        i += 1

class INQScheduler(object):
    def __init__(self, optimizer, iterative_steps):
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))
        if not iterative_steps[-1] == 1:
            raise ValueError("Last step should equal 1 in INQ.")
        self.optimizer = optimizer
        self.iterative_steps = iterative_steps
        self.idx = 0
        self.weight_bits = 5
        
        for group in self.optimizer.param_groups:
            group['ns'] = []
            group['Ts'] = []
            for p in group['params']:
                if p.requires_grad is False:
                    group['ns'].append((0, 0))
                    continue
                s = torch.max(torch.abs(p.data)).item()
                n_1 = math.floor(math.log((4*s)/3, 2))
                n_2 = int(n_1 + 1 - (2**(self.weight_bits-1))/2)
                group['ns'].append((n_1, n_2))

                
    #Quantize the parameters handled by the optimizer.
    def quantize(self):
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                T = group['Ts'][idx]
                ns = group['ns'][idx]
                device = p.data.device
                quantizer = partial(self.quantize_weight, n_1=ns[0], n_2=ns[1])
                fully_quantized = p.data.clone().cpu().apply_(quantizer).to(device)
                p.data = torch.where(T == 0, fully_quantized, p.data)
                p.data = fully_quantized
                
    #Quantize a single weight using INQ quantization scheme
    def quantize_weight(self, weight, n_1, n_2):
        alpha = 0
        beta = 2 ** n_2
        abs_weight = math.fabs(weight)
        quantized_weight = 0

        for i in range(n_2, n_1 + 1):
            if (abs_weight >= (alpha + beta) / 2) and abs_weight < (3*beta/2):
                quantized_weight = math.copysign(beta, weight)
            alpha = 2 ** i
            beta = 2 ** (i + 1)
        return quantized_weight
    
    #Quantize all parameters to integers
    def quantize_int(self):
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                ns = group['ns'][idx]
                device = p.data.device
                quantizer = partial(self.quantize_weight_int, n_1=ns[0], n_2=ns[1])
                fully_quantized = p.data.clone().cpu().apply_(quantizer).to(device)
                p.data = fully_quantized
                
    #Quantize a single weight to integer
    def quantize_weight_int(self, weight, n_1, n_2):
        alpha = 0
        beta = 2 ** n_2
        abs_weight = math.fabs(weight)
        quantized_weight = 0

        for i in range(n_2, n_1 + 1):
            if (abs_weight >= (alpha + beta) / 2) and abs_weight < (3*beta/2):
                quantized_weight = math.copysign(i, weight)
            alpha = 2 ** i
            beta = 2 ** (i + 1)
        return quantized_weight

    #Performs weight partitioning and quantization
    def step(self):
        for group in self.optimizer.param_groups:
            for idx, p in enumerate(group['params']):
                if p.requires_grad is False:
                    continue
                zeros = torch.zeros_like(p.data)
                ones = torch.ones_like(p.data)
                quantile = np.quantile(torch.abs(p.data.cpu()).numpy(), 1 - self.iterative_steps[self.idx])
                T = torch.where(torch.abs(p.data) >= quantile, zeros, ones)
                group['Ts'].append(T)
        self.idx += 1
        self.quantize()
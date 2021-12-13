

import torch
import numpy as np
import operator as op
import RBM.utils as utils
from torch.utils.data import Dataset



class AndDataset(Dataset):

    def __init__(self, size):
        """
        Args:
            size (string): size of artificial AND gate dataset.
        """
        self.size = size
        self.samples = [0]*size
        for i in range(size):
            A = i%2
            B = (i%4)>>1
            C = A & B
            self.samples[i] = torch.Tensor([A, B, C])


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.samples[index]


class OrDataset(Dataset):

    def __init__(self, size):
        """
        Args:
            size (string): size of artificial AND gate dataset.
        """
        self.size = size
        self.samples = [0]*size
        for i in range(size):
            A = i%2
            B = (i>>1)%2
            C = A | B
            self.samples[i] = torch.Tensor([A, B, C])


    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.samples[index]


"""
Implements the dataset for an arbitrary Dataset given a function that takes binary input values (i.e. 0s or 1s)

fun is a function that, given a Tensor of size (size, num_inputs) returns the desired output also as a Tensor

size should be a multiple of 2**num_inputs

random - True=use randomized samples, False=enumerate through all samples sequentially (should only bused if size is greater than 2**num_inputs)
prealloc - True=preallocate array in memory to make function calls faster. False=Each element is generated on the fly. Note: When using prealloc with random, results will be non-deterministic
"""
class FuncDataset(Dataset):

    def __init__(self, num_inputs, fun, size, random=False, prealloc=True, shuffle=False, block_size=-1):
        self.size = size
        self.num_inputs = num_inputs
        self.fun = fun
        self.prealloc = prealloc
        self.random = random
        self.block_size=block_size
        self.offset = 0
        self.shuffle = shuffle
        if block_size > 0:
            self.stride = block_size
        else:
            self.stride = size
        if prealloc:
            if random:
                inputs = torch.randint(0, 2, (self.stride, num_inputs))
            else:
                #Sequentially iterates through size (samples should be ordered)
                inputs = utils.intToTens(torch.arange(self.stride).float())[:, -1*num_inputs:]
                if inputs.size(1) < num_inputs:
                    inputs = torch.cat((torch.zeros(inputs.size(0), num_inputs - inputs.size(1)), inputs), dim = 1)

            self.samples = fun(inputs)
            if self.shuffle:
                self.samples = self.samples[torch.randperm(self.samples.size(0)), :]

    def __len__(self):
        return self.size
    def __getitem__(self, index):
        if self.prealloc and self.block_size < 0:
            return self.samples[index]
        elif self.prealloc:
            #Need to reinitialize samples
            if index < self.offset:
                self.offset = 0
            if index >= (self.offset + self.block_size):
                self.offset += self.block_size
                if self.random:
                    #Samples are random
                    inputs = torch.randint(0, 2, (self.stride, self.num_inputs))
                else:
                    #This creates sequential samples
                    inputs = utils.intToTens(torch.arange(self.offset, self.offset + self.block_size).float()[:, -1*self.num_inputs:])
                    #Padding the end with zeros if necessary
                    if inputs.size(1) < self.num_inputs:
                        inputs = torch.cat((torch.zeros(inputs.size(0), self.num_inputs - inputs.size(1)), inputs), dim = 1)
                #Find the correct output values
                #outputs = self.fun(inputs)
                #The samples are just the functions concatenated together
                #self.samples = torch.cat((inputs, outputs), dim=1)
                #Allows for shuffling of inputs and outputs as necessary
                self.samples = self.fun(inputs)
                if self.shuffle:
                    self.samples = self.samples[torch.randperm(self.samples.size(0)), :]

            return self.samples[index - self.offset]
        else:
            if self.random:
                inputs = torch.randint(0, 2, (1, self.num_inputs))[0]
                outputs = self.fun(inputs)
                return outputs
            else:
                sample = torch.zeros(self.num_inputs)
                sample = utils.intToTens(torch.Tensor([index]))[0]
                if len(sample.size()) == 0:
                    sample = torch.unsqueeze(sample, 0)
                sample = torch.cat((torch.zeros(self.num_inputs - sample.size(0)), sample), dim=0)[-1*self.num_inputs:]
                return self.fun(sample)

def _addWrapper(inputs):
    outputs = utils.tensbit_adder(inputs)
    return torch.cat((inputs, outputs), dim=1)

def _multWrapper(inputs):
    outputs = utils.tensbit_mult(inputs)
    return torch.cat((inputs, outputs), dim=1)

class AdderDataset(FuncDataset):
    def __init__(self, num_bits, size, random=False, prealloc=True, shuffle=False, block_size=-1):
        super(AdderDataset, self).__init__(2*num_bits + 1, _addWrapper, size, random=random, shuffle=shuffle, prealloc=prealloc, block_size=block_size)

class MultDataset(FuncDataset):
    def __init__(self, num_bits, size, random=False, prealloc=True, shuffle=False, block_size=-1):
        super(MultDataset, self).__init__(2*num_bits, _multWrapper, size, random=random, shuffle=shuffle, prealloc=prealloc, block_size=block_size)

if __name__ == "__main__":
    print("testing adder function")
    print("All should be [1, 0]")
    print(utils.bit_adder(0, 1, 1))
    print(utils.bit_adder(1, 0, 1))
    print(utils.bit_adder(1, 1, 0))
    print("All should be [0, 1]")
    print(utils.bit_adder(0, 0, 1))
    print(utils.bit_adder(0, 1, 0))
    print(utils.bit_adder(1, 0, 0))
    print("All should be [1, 1]")
    print(utils.bit_adder(1, 1, 1))
    print("\n")
    print("All should be [0, 1, 0]")

    FA = AdderDataset(1, 32)
    print("First 8 items of Full Adder dataset")
    for i in range(8):
        print(FA[i])
    FA2 = AdderDataset(2, 32)
    print("First 8 items of 2 bit adder dataset")
    for i in range(8):
        print(FA2[i])
    FA3 = AdderDataset(3, 32)
    print("First 8 itmes of 3 bit adder dataset")
    for i in range(8):
        print(FA3[i])

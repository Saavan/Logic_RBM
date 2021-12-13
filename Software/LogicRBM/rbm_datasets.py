import os
import sys
import utils
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RBM.rbm_datasets import *

def addWrapper(inputs):
    outputs = utils.tensbit_adder(inputs)
    return torch.cat((inputs.float(), outputs.float()), dim=1)

def multWrapper(inputs):
    outputs = utils.tensbit_mult(inputs)
    return torch.cat((inputs.float(), outputs.float()), dim=1)



def MergeMultFunc(inputs, mult_func, add_func, multbits=None, addbits=None):
    '''
    This is the input function for the rbm_datasets class
    '''
    if not inputs.size(1)%4 == 0:
        raise ValueError("Need even number of inputs and even number of num_bits")

    num_bits = int(inputs.size(1)/2)
    half_bits = int(num_bits/2)

    if multbits is None:
        multbits = torch.arange(4*half_bits, dtype=torch.long)
    if addbits is None:
        addbits = torch.arange(3*half_bits+2, dtype=torch.long)


    A1 = inputs[:, 0:half_bits]
    A0 = inputs[:, half_bits:2*half_bits]
    B1 = inputs[:, 2*half_bits:3*half_bits]
    B0 = inputs[:, 3*half_bits:4*half_bits]

    A0B0 = mult_func(torch.cat((A0, B0), dim=1))

    A0B1 = mult_func(torch.cat((A0, B1), dim=1))

    A1B0 = mult_func(torch.cat((A1, B0), dim=1))

    A1B1 = mult_func(torch.cat((A1, B1), dim=1))

    FA0_in = torch.cat((torch.zeros(inputs.size(0), 1), A0B0[:, multbits[2*half_bits:3*half_bits]], A0B1[:, multbits[3*half_bits:]]), dim=1)
    FA0 = add_func(FA0_in)

    FA1_in = torch.cat((torch.zeros(inputs.size(0), 1), FA0[:, addbits[2*half_bits+2:3*half_bits+2]], A1B0[:, multbits[3*half_bits:]]), dim=1)
    FA1 = add_func(FA1_in)

    FA2_in = torch.cat((FA0[:, addbits[2*half_bits+1:2*half_bits+2]], A0B1[:,  multbits[2*half_bits:3*half_bits]], A1B0[:, multbits[2*half_bits:3*half_bits]]), dim=1)
    FA2 = add_func(FA2_in)

    FA3_in = torch.cat((FA1[:, addbits[2*half_bits+1:2*half_bits+2]], FA2[:, addbits[2*half_bits+2:3*half_bits+2]] , A1B1[:, multbits[3*half_bits:]]), dim=1)
    FA3 = add_func(FA3_in)

    FA4_in = torch.cat((FA2[:, addbits[2*half_bits+1:2*half_bits+2]], torch.zeros(inputs.size(0), half_bits-1), FA3[:, addbits[2*half_bits+1:2*half_bits+2]],
                       A1B1[:, multbits[2*half_bits:3*half_bits]]), dim=1)

    FA4 = add_func(FA4_in)
    outs = torch.zeros(inputs.size(0), 9*num_bits+6)


    A0B0_out = A0B0

    multbits_out = torch.ones(A0B0.size(1)).byte()
    multbits_out[multbits[:half_bits]] = 0
    A0B1_out = A0B1[:, multbits_out]

    multbits_out = torch.ones(A0B0.size(1)).byte()
    multbits_out[multbits[half_bits:2*half_bits]] = 0
    A1B0_out = A1B0[:, multbits_out]

    multbits_out = torch.ones(A0B0.size(1)).byte()
    multbits_out[multbits[:2*half_bits]] = 0
    A1B1_out = A1B1[:, multbits_out]


    addbits_out = torch.ones(FA0.size(1)).byte()
    addbits_out[addbits[:2*half_bits+1]] = 0
    FA0_out = FA0[:, addbits_out]
    FA1_out = FA1[:, addbits_out]
    FA2_out = FA2[:, addbits_out]
    FA3_out = FA3[:, addbits_out]
    FA4_out = FA4[:, addbits_out]

    outs = torch.cat((A0B0_out, A0B1_out, A1B0_out, A1B1_out, torch.zeros(inputs.size(0), 1),
                     FA0_out, torch.zeros(inputs.size(0), 1), FA1_out, FA2_out, FA3_out,
                     torch.zeros(inputs.size(0), half_bits-1), FA4_out), dim=1)
    return outs




class MergeMultDataset(FuncDataset):
    def __init__(self, num_bits, size, random=False, prealloc=True, shuffle=False, block_size=-1):
        super(MergeMultDataset, self).__init__(2*num_bits, lambda x: MergeMultFunc(x, multWrapper, addWrapper),
                size, random=random, shuffle=shuffle, prealloc=prealloc, block_size=block_size)


class MergeMult16Dataset(FuncDataset):
    def __init__(self, num_bits, size, random=False, prealloc=True, shuffle=False, block_size=-1):
        Mult8_outbits = torch.LongTensor([28, 29, 30, 31,  0,  1,  2,  3, 16, 17, 18, 19,  4,  5,  6,  7, 74, 75,
        76, 77, 66, 67, 68, 69, 56, 57, 58, 59, 12, 13, 14, 15])
        super(MergeMult16Dataset, self).__init__(2*num_bits, lambda y: MergeMultFunc(y, lambda x: MergeMultFunc(x, multWrapper, addWrapper),
            addWrapper, multbits=Mult8_outbits), size, random=random, shuffle=shuffle, prealloc=prealloc, block_size=block_size)


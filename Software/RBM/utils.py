import numpy as np
import torch
import matplotlib.pyplot as plt
import psutil
import os
import gc
import sys

"""
Euclidean distance between two discrete probability distributions. Note both distributions should be the same size
p1 - first probability distribution
p2 - second distribution
return float representing the sum of the euclidean distances
"""
def EuDist(p1, p2):
    if not len(p1) == len(p2):
        raise ValueError("The two probability distributions are not the same size!")
    return sum([(p1[i] - p2[i])**2 for i in range(len(p1))])


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """

    return torch.sum(torch.where(p != 0, p * torch.log(p / q), torch.zeros(p.size(0))))

def acf(x, length=20):
    """Autocorrelation function for an input vector x. Note this normalizes with respect
    to variance, so vectors that are very slow moving will return NaN
    x - vector to calculate autocorrelation of
    length - length to calculate autocorrelation coefficient out to
     """
    return np.array([1]+[np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, length)])

"""
Maximum absolute element-wise difference between p and q, i.e. linf(p - q)
"""
def variation_distance(p, q):
    return torch.max(torch.abs(p - q))

"""
Infinity norm of p, i.e. maximum absolute value of p's elements
"""
def linf(p):
    return torch.max(torch.abs(p))

"""
Generates an ideal distribution based on a variable length input and output function. All of the input and output functions
should be bitwise, i.e. accept 1 or 0 as its inputs and outputs
num_inputs - Number of inputs to the function provided
num_outputs - number of outputs the function provides
fun - bitwise function with the above number of inputs and outputs. Output of fun should be a list of bits i.e. [1, 1, 0]
returns - a list of values which are 1/total if the input output pair is a possible answer to the input function and 0 otherwise. total is the total number of answers the input function has.
"""
def generateDistribution(num_inputs, num_outputs, fun):
    corr_probs = np.array([0] * (2**(num_inputs+num_outputs)))
    for x in range(2**num_inputs):
        key = bin(x)[2:]
        key = '0'*((num_inputs)-len(key)) + key
        outval = fun(*[int(x) for x in key])
        out = ''.join([str(x) for x in outval])
        val = key + out
        corr_probs[int(val, 2)] = 1
    tot = sum(corr_probs)
    corr_probs = corr_probs / tot
    return corr_probs

def adderDistribution(num_bits):
    return generateDistribution(2*num_bits + 1, num_bits+1, bit_adder)

def multDistribution(num_bits):
    return generateDistribution(2*num_bits, 2*num_bits, bit_mult)

"""
Plots the probabilities in a new pyplot figure. Has a bar graph and legend.
keys - Labels for the x axis
arr - array of arrays, each array representing a discrete probability distribution
labels - legend labels each of the arrays
"""
def plotProbs(keys, arr, labels=None):
    fig = plt.figure()
    x = np.arange(len(keys))
    width = 1/(1+len(arr))
    for i,vals in enumerate(arr):
        if labels:
            plt.bar(x + i*width - (len(arr)-1)*width/2, vals, width, label=labels[i])
        else:
            plt.bar(x + i*width - (len(arr)-1)*width/2, vals, width)
    plt.xticks(x, keys)
    if(labels):
        plt.legend()
    return fig

"""
Marginalizes the probability distribution with the given output bits. The outbits should be given in the order
they wish to be outputted in.
probs - dictionary of key value pairs, keys representing indices and values representing that value's probability
        this should be in the same format as generate_statistics makes
outbits - list of bits that are NOT marginalized over, i.e. bits that are kept as visible
"""
def marginalize(probs, outbits):
    out = {}
    for key in probs:
        newKey = ''.join(key[i] for i in outbits)
        if newKey in out:
            out[newKey] += probs[key]
        else:
            out[newKey] = probs[key]
    return out

"""
this funcion serves as an arbitrary length bitwise adder
Cin - Carry in
*args - should be even number of single bits organized in big-endian format, i.e. args = [A1, A0, B1, B0] for a two bit adder or args = [A2, A1, A0, B2, B1, B0] for a 3 bit adder.
returns - a list of bits containing the output, in big-endian form, i.e. [out2, out1, out0]
"""
def bit_adder(Cin, *args):
    if len(args) < 2:
        raise ValueError("Not enough arguments to adder!")
    if not len(args) % 2 == 0:
        raise ValueError("Should be odd number of arguments!")
    A = args[:int(len(args)/2)]
    B = args[int(len(args)/2):]
    A = int(''.join([str(int(x)) for x in A]), 2)
    B = int(''.join([str(int(x)) for x in B]), 2)
    S = A + B + int(Cin)
    out = bin(S)[2:]
    out = '0' * (int(len(args)/2) - len(out) + 1) + out
    return list(map(int, out))

"""
Does the same thing as bit_adder, but using tensors. Takes in a tensor of size(num_examples, 2*num_bits+1)
and returns the bitwise addition along standard format (i.e. [A1, A0, B1, B0] see above).
Doesn't suffer from overflow and is (fairly) quick. Still slower than using listtoInt and then doing regular addition, but this version works for arbitrary length vectors.
"""
def tensbit_adder(tens):
    if len(tens.size()) == 1:
        return tens.new_tensor(bit_adder(int(tens[0]), *tens[1:].long().numpy()))
    if (tens.size(1)) < 2:
        raise ValueError("Not enough arguments to adder!")
    if not tens.size(1) % 2 == 1:
        raise ValueError("Should be odd number of arguments!")
    num_bits = int((tens.size(1) - 1)/2)
    mid = num_bits + 1

    out = tens[:, 1:mid] + tens[:, mid:]
    out[:, -1] += tens[:, 0]

    #Appending leading zero
    out = torch.cat((torch.zeros(out.size(0), 1, device=out.device), out), dim=1)

    return _resid(out)


"""
this function serves as an arbitrary length bitwise multiplier
*args - should be even number of single bits organized in big-endian format, i.e. args = [A1, A0, B1, B0] for a two bit multiplier or args = [A2, A1, A0, B2, B1, B0] for a 3 bit multiplier.
returns - a list of bits containing the output, in big-endian form, i.e. [out2, out1, out0]
"""
def bit_mult(*args):
    if len(args) < 2:
        raise ValueError("Not enough arguments to multiplier!")
    if not len(args) % 2 == 0:
        raise ValueError("Should be even number of arguments!")
    A = args[:int(len(args)/2)]
    B = args[int(len(args)/2):]
    A = int(''.join([str(int(x)) for x in A]), 2)
    B = int(''.join([str(int(x)) for x in B]), 2)
    S = A * B
    out = bin(S)[2:]
    out = '0' * ((len(args)) - len(out)) + out
    return list(map(int, out))

"""
Does the same thing as bit_mult, but using tensors, making it faster. Takes in a tensor of size(num_examples, 2*num_bits) and returns the bitwise addition along standard format (i.e. [A1, A0, B1, B0] see above).
This function works for arbitrary number of bits (just as bit_mult does). Also doesn't convert back and forth between tensor and python formats making it work with CUDA.
"""
def tensbit_mult(tens):
    if len(tens.size()) == 1:
        return tens.new_tensor(bit_mult(*tens.long().numpy()))
    if (tens.size(1)) < 2:
        raise ValueError("Not enough arguments to adder!")
    if not tens.size(1) % 2 == 0:
        raise ValueError("Should be even number of arguments!")

    num_bits = int(tens.size(1)/2)
    tens1 = tens[:, :num_bits].float()
    tens2 = tens[:, num_bits:].float()
    out = torch.zeros(tens1.size(0), tens1.size(1) * 2, device=tens.device).float()
    offset = 1
    #Performing multiplication (shift, multiply, add)
    for val in tens2.t():
        out[:, offset:tens1.size(1)+offset] += tens1 * val.unsqueeze(1)
        offset += 1

    return _resid(out)


"""
Helper function to find residuals and convert to a correct 2 bit representation from a pseudo-2 bit that has all of the carries accumulated into one number
"""
def _resid(tens):
    out = tens
    mod = out % 2
    resid = torch.nonzero(out - mod)
    while not torch.equal(mod, out):
        resid = resid[torch.ge(out[resid[:, 0], resid[:, 1]] , 2)]
        carry = (out[resid[:, 0], resid[:, 1]] / 2).floor()
        out = mod
        resid[:, 1] = resid[:, 1] - 1
        out[resid[:, 0], resid[:, 1]] += carry
        mod = out % 2
    return out

"""
Combines multiple probability distributions that would have been generated by RBM.generate_statistics or _gen_stat_helper
"""
def combine_probs(a, b):
    return {**a, **b, **{k:a[k] +  b[k] for k in set(b) & set(a)}}


"""
Takes a number and converts it to a bit tensor of 0s and 1s.
"""
def num2bits(x):
    return torch.Tensor(list(map(int, bin(x)[2:])))

"""
Returns a list of d-length lists which are all permutations of binary numbers of length d
Size of returned list is 2**d
"""
def binary_permutations(d):
    return [[int(e) for e in bin(i)[2:].zfill(d)] for i in range(2**d)]

"""
Converts a tensor filled with 0s and 1s to a string of that tensor.
Stride is needed to convert tensors of length greater than 32 (which cause overflow) into a string
"""
def tensorToString(x, stride=16):
    out = ''
    ind = 0
    pows = torch.pow(2, x.new_tensor(np.flip(np.arange(stride), axis=0).copy())).float()
    while ind <= (x.size(0) - stride):
        curr = ind
        nex = ind+stride
        temp = bin(int(torch.sum(pows*x[ind:ind+stride])))[2:]
        out += '0'*(stride - len(temp)) + temp
        ind = ind + stride

    nex = min(stride, x.size(0) - ind)
    pows = torch.pow(2, x.new_tensor(np.flip(np.arange(min(stride, nex)), axis=0).copy())).float()
    temp = bin(int(torch.sum(pows*x[ind:])))[2:]
    out += '0'*(nex - len(temp)) + temp
    return '0'*(x.size(0) - len(out)) + out

"""
Converts from an numpy buffer (just a byte string containing the values) to a tensor object
Used in tensgenerate_statistics to convert the keys back from string.
"""
def fromBuffer(x):
    return torch.tensor(np.frombuffer(x, dtype=np.uint8), dtype=torch.float32)


"""
Converts list of binary values (works with any iterable) to a single integer using an accumulator. This seems to run faster than using map?
This function also suffers from floating point overflow problems associated with torch using float32 and float64 values.
"""
def listToInt(nums):
    if type(nums) == torch.Tensor and nums.dtype == torch.float32 and nums.size(0) >= 25:
        raise ValueError("Tried to convert a greater than 32 bit number in torch! Convert to float64 before using")
    if type(nums) == torch.Tensor and nums.dtype == torch.float64 and nums.size(0) >= 54:
        raise ValueError("Tried to convert a greater than 64 bit number in torch! Not supported")
    tot = 0
    for num in nums:
        tot *= 2
        tot += num
    return tot



"""
Takes a tensor full of ints, and converts each of them into their bitwise representation.

"""
def intToTens(tens):
    outlist = []
    temp = tens.clone().detach().float()
    #Degenerate case
    if torch.equal(tens, torch.zeros_like(tens)):
        return torch.zeros_like(tens)

    while not torch.equal(temp.float(), torch.zeros(temp.size(0), device = temp.device).float()):
        outlist.append(temp % 2)
        temp = (temp/2).floor()
    out = torch.stack(tuple(reversed(outlist)), dim=1)
    return out

"""
Takes standard adder input bits, converts to a printable string
"""
def AdderToString(tens):
    num_bits = int((tens.size(0) - 2)/3)
    if not tens.size(0) == 3*num_bits+2:
        raise TypeError('Size is not a valid adder size!')
    return "{0} +  {1} + {2} = {3}".format(int(tens[0]), listToInt(tens[1:num_bits+1]), listToInt(tens[num_bits+1:2*num_bits+1]), listToInt(tens[2*num_bits+1:]))

def MultToString(tens):
    num_bits = int(len(tens)/4)
    if not len(tens) == 4*num_bits:
        raise TypeError('Size is not a valid multiplier size!')
    return "{0} * {1} = {2}".format(listToInt(tens[:num_bits]), listToInt(tens[num_bits:2*num_bits]), listToInt(tens[2*num_bits:]))

'''
Takes a standard mult input bits (as outputted by tensgenerate_statistics) and converts to the values they represent
'''
def MultToVals(tens):
    num_bits = int(len(tens)/4)
    if not len(tens) == 4*num_bits:
        raise TypeError('Size is not a valid multiplier size!')
    return (listToInt(tens[:num_bits]), listToInt(tens[num_bits:2*num_bits]), listToInt(tens[2*num_bits:]))

"""
Checks current memory usage of the process. Useful for debugging memory leaks and such.
"""
def mem_test():
    process = psutil.Process(os.getpid())
    print("Space (Gb) occupied by Process", (process.memory_info().rss)/(2**30))

    num_objs = 0
    num_floats = 0
    space = 0
    xyz = 1
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #print(type(obj), obj.size())
                num_floats += obj.nelement()
                space += obj.element_size() * obj.nelement()
                num_objs += 1
        except:
            xyz += 1


# Given a bit string repr,
# performs a bit flip on all bits
def bit_flip(bit_str):
    result = ''
    for bit in bit_str:
        if bit == '0':
            result += '1'
        else:
            result += '0'
    return result

# Given a bit string, returns 2's complement
# Source: https://www.geeksforgeeks.org/1s-2s-complement-binary-number/
def twos_comp(bit_str):
    flipped = list(bit_flip(bit_str))
    result = flipped
    n = len(bit_str)
    for i in range(n - 1, -1, -1):
        if (flipped[i] == '1'):
            result[i] = '0'
        else:
            result[i] = '1'
            break
    return ''.join(result)


# Given a string "number_str" of size n bits,
# and location of binary point "p" bits from the LSB,
# convert it to decimal value
# Ex: [n] [n - 1] ---- [p+1][p].[p-1]....[1][0]
# Note that the point is not passed in this string

def fixed_to_decimal (number_str, p):
    #Check if negative
    is_neg = 0;
    if number_str[0] == '1':
        is_neg = 1
        number_str = twos_comp(number_str)
    # Perform conversion
    result = 0;
    n = len(number_str)
    for i in range (n):
        if number_str[i] == '1':
            result += pow(-1, is_neg)*pow(2, n - 1 - i - p)
    return result

# Given a decimal number "number_dec" in python's floating point,
# and a target "n" bits to convert into a fixed point binary string
# whose point is at indes "p", return the "number_str" representation
# as well as the residual
def decimal_to_fixed(number_dec, n, p):
    #Check if negative
    result = '0';
    is_neg = 0
    if number_dec < 0:
        is_neg = 1
        number_dec = -1*number_dec

    #Perform conversion for positive
    residual = number_dec
    lsb = pow(2, -p)
    for i in range (n - 1):
        bit_val = pow(2, n - 2 - i - p)
        if  bit_val <= residual + lsb/2:
            result += '1'
            residual -= bit_val
        else:
            result += '0'

    if is_neg:
        result = twos_comp(result)
    return [result, residual]


def quantize_tens(x, n, p):
    '''Takes a tensor and returns a quantized version of this. This is CUDA safe'''
    out = torch.round(x * 2**p) / 2**(p)
    out = torch.clamp(out, max=(2**(n-p-1) - 2**(-p)), min=-1*(2**(n-p-1) - 2**(-p)))
    return out

def quant_val(x, n, p):
    '''Take an input value and returns a quantized versin of it'''
    out = np.round(x * 2**p) / 2**(p)
    out = min(out, (2**(n-p-1) - 2**(-p)))
    out = max(out, -1*(2**(n-p-1) - 2**(-p)))
    return out

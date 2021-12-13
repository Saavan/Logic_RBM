import os
import sys
import torch
import copy
import time
import re
sys.path.insert(0,  '../')
from RBM.rbm import *
import RBM.utils as utils





def cnf2rbm(file_name, lambd=1, amp=20):

    #initializing models
    dataNOT = torch.Tensor([[1, 0], [0, 1]])
    dataHalfOR = torch.Tensor([[int(x) for x in '{:0>2}'.format(bin(y)[2:])] for y in range(1, 2**2)])
    data3HalfOR = torch.Tensor([[int(x) for x in '{:0>3}'.format(bin(y)[2:])] for y in range(1, 2**3)])
    data4HalfOR = torch.Tensor([[int(x) for x in '{:0>4}'.format(bin(y)[2:])] for y in range(1, 2**4)])

    n = RBM.populate_model(amp, lambd, dataNOT)
    oh = RBM.populate_model(amp, lambd, dataHalfOR)
    o3h = RBM.populate_model(amp, lambd, data3HalfOR)
    o4h = RBM.populate_model(amp, lambd, data4HalfOR)
    o_models = [oh, o3h, o4h]

    f = open(file_name)
    ind = 0
    raw_clauses = []
    for line in f.readlines():
        if line[0] == 'c':
            print(line,  end='')
            continue
        if line[0] == 'p':
            temp = [int(s) for s in re.findall("[-\d]+", line)]
            num_vars = temp[0]
            num_clauses = temp[1]
            clauses = [[] for _ in range(num_vars)]
            nclauses = [[] for _ in range(num_vars)]
            print('vars:', num_vars, 'clauses:', num_clauses)
            continue
        #Parses out integers from the
        out = [int(s) for s in re.findall("[-\d]+", line)]
        raw_clauses.append(out)
        if ind == 0:
            #Initializing the model
            model = o_models[len(out) - 3]
        else:
            #Merging more clauses onto the model, depending on size of clause
            model = RBM.merge(model, o_models[len(out) - 3], [])
        for var in out:
            if var > 0:
                clauses[abs(var)-1].append(ind)
            if var < 0:
                nclauses[abs(var)-1].append(ind)
            if not var == 0:
                ind+=1


    #Connecting NOT gates
    for var, nvar in zip(clauses, nclauses):
        if not (var == [] or nvar == []):
            model = RBM.merge(model, n, [(var[0], 0), (nvar[0], 1)])

    #Collapsing redundant nodes
    temp = []
    for var, nvar in zip(clauses, nclauses):
        temp = temp + list(zip([var[0] for _ in range(len(var)-1)], var[1:]))
        temp = temp + list(zip([nvar[0] for _ in range(len(nvar)-1)], nvar[1:]))
    print(temp)
    model.collapse(temp)
    model.fname = file_name
    model.clauses = (clauses, nclauses)
    model.outbits = torch.LongTensor([var[0] for var in clauses])
    model.raw_clauses = raw_clauses

    return model

if __name__ == "__main__":
    model = cnf2rbm('benchmarks/simple_v3_c4.cnf')
    print(model.outbits)
    probs = model.probs().numpy()
    sort = np.flip(np.argsort(probs), 0)
    print(sort[:5])
    print(probs)
    print(model.num_visible)
    print(probs[sort[:8]], [bin(x)[2:] for x in sort[:8]])

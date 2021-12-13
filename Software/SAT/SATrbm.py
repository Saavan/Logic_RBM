import os
import sys
import torch
import copy
import time
import re
sys.path.insert(0,  '../')
from RBM.rbm import *
import RBM.utils as utils


class SATrbm(RBM):
    def __init__(self, fname, lambd=1, amp=20):
        #initializing gates
        dataNOT = torch.Tensor([[1, 0], [0, 1]])
        dataHalfOR = torch.Tensor([[int(x) for x in '{:0>2}'.format(bin(y)[2:])] for y in range(1, 2**2)])
        data3HalfOR = torch.Tensor([[int(x) for x in '{:0>3}'.format(bin(y)[2:])] for y in range(1, 2**3)])
        data4HalfOR = torch.Tensor([[int(x) for x in '{:0>4}'.format(bin(y)[2:])] for y in range(1, 2**4)])

        n = RBM.populate_model(amp, lambd, dataNOT)
        o2h = RBM.populate_model(amp, lambd, dataHalfOR)
        o3h = RBM.populate_model(amp, lambd, data3HalfOR)
        o4h = RBM.populate_model(amp, lambd, data4HalfOR)
        o_models = [o2h, o3h, o4h]

        f = open(fname)
        ind = 0
        raw_clauses = []
        self.num_vars = -1
        self.num_clauses = -1
        for line in f.readlines():
            if line[0] == 'c':
                print(line,  end='')
                continue
            if line[0] == 'p':
                temp = [int(s) for s in re.findall("[-\d]+", line)]
                num_vars = temp[0]
                self.num_vars = num_vars
                num_clauses = temp[1]
                self.num_clauses = num_clauses
                clauses = [[] for _ in range(num_vars)]
                nclauses = [[] for _ in range(num_vars)]
                print('vars:', num_vars, 'clauses:', num_clauses)
                continue
            if num_vars == -1 or num_clauses == -1:
                raise ValueError('No valid configuration given in CNF file! Need line starting with p')
            #Parses out integers from the
            out = [int(s) for s in re.findall("[-\d]+", line)]
            # print(out, len(raw_clauses))
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

        # print(model.num_visible)
        #Connecting NOT gates
        for var, nvar in zip(clauses, nclauses):
            if not (var == [] or nvar == []):
                model = RBM.merge(model, n, [(var[0], 0), (nvar[0], 1)])

        for i, var in enumerate(clauses):
            print('var:', i+1, 'inds:', var)
        model.outbits = torch.LongTensor([var[0] for var in clauses])
        print('intial setting of outbits', model.outbits, 'len:', len(model.outbits))
        #Collapsing redundant nodes
        temp = []
        for var, nvar in zip(clauses, nclauses):
            temp = temp + list(zip([var[0] for _ in range(len(var)-1)], var[1:]))
            temp = temp + list(zip([nvar[0] for _ in range(len(nvar)-1)], nvar[1:]))
        # print(temp)
        model.collapse(temp)

        #Transferring

        self.fname = fname
        self.clauses = (clauses, nclauses)
        self.raw_clauses = raw_clauses
        #Call Base classes init method
        RBM.__init__(self, model.num_visible, model.num_hidden, 1, outbits=model.outbits)
        #Transfer over the weights and visible/hidden bias
        self.weights = model.weights
        self.visible_bias = model.visible_bias
        self.hidden_bias = model.hidden_bias


    def satisfied_clauses(self, cand):
        '''
        Returns the number of satisfied clauses that the candidate solution works for
        '''
        if not len(cand) == self.num_vars:
            raise ValueError('Candidate solution should have same number of variables')
        sats = 0
        for clause in self.raw_clauses:
            #Creates the statement to be OR-d together
            vals = [cand[ind - 1] if ind>0 else (not cand[abs(ind) - 1]) for ind in clause[:-1]]
            #If any of values is true, this is equivalent to an OR statement
            sats += any(vals)
        return sats

if __name__ == "__main__":
    model = SATrbm('benchmarks/simple_v3_c2.cnf')
    print(model.outbits)
    probs = model.probs().numpy()
    sort = np.flip(np.argsort(probs), 0)
    print(sort[:5])
    print(probs)
    print(model.num_visible)
    print(probs[sort[:8]], ['{:0>5}'.format(bin(x)[2:]) for x in sort[:8]])
    for test in ['{:0>5}'.format(bin(x)[2:]) for x in sort[:8]]:
        temp = torch.LongTensor(list(map(int, test)))
        print(temp)
        print([int(x) for x in [test[int(x)] for x in model.outbits]])
        print('satisfied clauses:', model.satisfied_clauses([int(x) for x in temp[model.outbits]]))

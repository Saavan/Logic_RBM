"""
This file includes tests for various RBM functions.
"""

import numpy as np
import torch
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import RBM.utils as utils

def test_add(adderRBM, num_cases, CDk, samps, debug=False):

    corr = 0

    inp = torch.randint(0, 2, (num_cases, 2*adderRBM.num_bits+1), device=torch.device(adderRBM.device)).float()
    out = utils.tensbit_adder(inp)
    clamp = torch.zeros(num_cases, 3*adderRBM.num_bits+2, device=torch.device(adderRBM.device)) - 1
    clamp[:, :2*adderRBM.num_bits+1] = inp
    outList = adderRBM.tensgenerate_statistics(samps, num_cases, k=CDk, clamp=clamp)[0]

    ans = torch.cat((inp, out), dim=1).cpu()

    for i, sampDict in enumerate(outList):
        v=list(sampDict.items())
        vals = [x[1] for x in v]
        MLE = v[np.argmax(vals)]
        MLE = (utils.fromBuffer(MLE[0]), MLE[1])

        if torch.equal(ans[i], MLE[0]):
            if debug:
                print("Correct!\n")
            corr += 1
    return corr


def test_rev(adderRBM, num_cases, CDk, samps, debug=False):

    corr = 0

    inp = torch.randint(0, 2, (num_cases, 2*adderRBM.num_bits+1), device=torch.device(adderRBM.device)).float()
    out = utils.tensbit_adder(inp)
    clamp = torch.zeros(num_cases, 3*adderRBM.num_bits+2, device=torch.device(adderRBM.device)) - 1
    clamp[:, 2*adderRBM.num_bits+1:3*adderRBM.num_bits+2] = out
    outList = adderRBM.tensgenerate_statistics(samps, num_cases, k=CDk, clamp=clamp)[0]

    ans = torch.cat((inp, out), dim=1).cpu()
    out = out.cpu()

    for i, sampDict in enumerate(outList):
        v=list(sampDict.items())
        vals = [x[1] for x in v]
        MLE = v[np.argmax(vals)]
        MLE = (utils.fromBuffer(MLE[0]), MLE[1])

        sampAns = torch.Tensor(utils.bit_adder(*(MLE[0][:2*adderRBM.num_bits+1])), device=torch.device(adderRBM.device))

        if torch.equal(out[i], sampAns):
            if debug:
                print("Correct!\n")
            corr += 1
    return corr



def test_sub(adderRBM, num_cases, CDk, samps, debug=False):

    corr = 0

    inp = torch.randint(0, 2, (num_cases, 2*adderRBM.num_bits+1), device=torch.device(adderRBM.device)).float()
    out = utils.tensbit_adder(inp)
    clamp = torch.zeros(num_cases, 3*adderRBM.num_bits+2, device=torch.device(adderRBM.device)) - 1
    clamp[:, :adderRBM.num_bits+1] = inp[:, :adderRBM.num_bits+1]
    clamp[:, 2*adderRBM.num_bits+1:3*adderRBM.num_bits+2] = out
    outList = adderRBM.tensgenerate_statistics(samps, num_cases, k=CDk, clamp=clamp)[0]

    ans = torch.cat((inp, out), dim=1).cpu()

    for i, sampDict in enumerate(outList):
        v=list(sampDict.items())
        vals = [x[1] for x in v]
        MLE = v[np.argmax(vals)]
        MLE = (utils.fromBuffer(MLE[0]), MLE[1])

        if torch.equal(ans[i], MLE[0]):
            if debug:
                print("Correct!\n")
            corr += 1
    return corr



def test_mult(multRBM, num_cases, CDk, samps, debug=False):
    corr = 0

    inp = torch.randint(0, 2, (num_cases, 2*multRBM.num_bits), device=torch.device(multRBM.device)).float()
    out = utils.tensbit_mult(inp)
    clamp = torch.zeros(num_cases, 4*multRBM.num_bits, device=torch.device(multRBM.device)) - 1
    clamp[:, :2*multRBM.num_bits] = inp
    outList = multRBM.tensgenerate_statistics(samps, num_cases, k=CDk, clamp=clamp)[0]

    ans = torch.cat((inp, out), dim=1).cpu()

    for i, sampDict in enumerate(outList):
        v=list(sampDict.items())
        vals = [x[1] for x in v]
        MLE = v[np.argmax(vals)]
        MLE = (utils.fromBuffer(MLE[0]), MLE[1])

        if debug:
            MLE_printer = [y.item() for y in MLE[0]]
            ans_printer = [y.item() for y in ans[i]]
            print(utils.MultToString(MLE_printer), MLE[1])
            print(utils.MultToString(ans_printer))

        if torch.equal(ans[i], MLE[0]):
            if debug:
                print("Correct!\n")
            corr += 1
    return corr


def test_div(multRBM, num_cases, CDk, samps, debug=False):
    corr = 0


    inp = torch.randint(0, 2, (num_cases, 2*multRBM.num_bits), device=torch.device(multRBM.device)).float()
    out = utils.tensbit_mult(inp)
    clamp = torch.zeros(num_cases, 4*multRBM.num_bits, device=torch.device(multRBM.device)) - 1
    clamp[:, :multRBM.num_bits] = inp[:, :multRBM.num_bits]
    clamp[:, 2*multRBM.num_bits:] = out
    outList = multRBM.tensgenerate_statistics(samps, num_cases, k=CDk, clamp=clamp)[0]

    ans = torch.cat((inp, out), dim=1).cpu()
    corrAns = out.cpu()

    for i, sampDict in enumerate(outList):
        v=list(sampDict.items())
        vals = [x[1] for x in v]
        MLE = v[np.argmax(vals)]
        MLE = (utils.fromBuffer(MLE[0]), MLE[1])

        sampAns = torch.Tensor(utils.bit_mult(*(MLE[0][:2*multRBM.num_bits])))

        if debug:
            MLE_printer = [y.item() for y in MLE[0]]
            ans_printer = [y.item() for y in ans[i]]
            print(utils.MultToString(MLE_printer), MLE[1])
            print(utils.MultToString(ans_printer))

        if torch.equal(corrAns[i], sampAns):
            if debug:
                print("Correct!\n")
            corr += 1
    return corr


def test_fact(multRBM, num_cases, CDk, samps, debug=False):
    corr = 0

    inp = torch.randint(0, 2, (num_cases, 2*multRBM.num_bits), device=torch.device(multRBM.device)).float()
    out = utils.tensbit_mult(inp)
    clamp = torch.zeros(num_cases, 4*multRBM.num_bits, device=torch.device(multRBM.device)) - 1
    clamp[:, 2*multRBM.num_bits:] = out
    outList = multRBM.tensgenerate_statistics(samps, num_cases, k=CDk, clamp=clamp)[0]

    corrAns = out.cpu()

    for i, sampDict in enumerate(outList):
        v=list(sampDict.items())
        vals = [x[1] for x in v]
        MLE = v[np.argmax(vals)]
        MLE = (utils.fromBuffer(MLE[0]), MLE[1])

        sampAns = torch.Tensor(utils.bit_mult(*(MLE[0][:2*multRBM.num_bits])))

        if debug:
            MLE_printer = [y.item() for y in MLE[0]]
            print(utils.MultToString(MLE_printer), MLE[1])

        if torch.equal(corrAns[i], sampAns):
            if debug:
                print("Correct!\n")
            corr += 1
    return corr

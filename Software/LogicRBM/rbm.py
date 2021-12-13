import os
import sys
import copy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from RBM.rbm import *


class AdderRBM(RBM):

    """
    """
    def __init__(self, num_bits, num_hidden, k, learning_rate=1e-3,  use_momentum=False, momentum_coefficient=0.5, decay_type='L2', weight_decay=1e-4, device='cpu'):
        super(AdderRBM, self).__init__(3*num_bits+2, num_hidden, k, learning_rate=learning_rate, use_momentum=use_momentum, momentum_coefficient=momentum_coefficient, decay_type=decay_type, weight_decay=weight_decay, device=device, outbits=None)
        self.num_bits=num_bits


    """
    Merges two adders into a larger adder that is the sum of the two adder bits. Training parameters are taken from the first adder class. The least significant bits come from adder1, the most significant bits from adder2.
    Assumes the outbits for each adder are of the form [Cin, A1, A0, B1, B0, Cout, S1, S0] (2 bit adder), [Cin, A0, B0, Cout, S0] (1 bit adder) etc.
    """
    def merge_adders(adder1, adder2):
        if not (type(adder1) == AdderRBM and type(adder2) == AdderRBM):
            raise TypeError("Both adders should be of type AdderRBM!")
        out = RBM.merge(adder1, adder2, [(adder1.outbits[2*adder1.num_bits+1], adder2.outbits[0])])

        offset = len(adder1.outbits)

        #Expected starting format for outbits (merging 1 bit adders into 2 bit as example)
        #[Cin, A0, B0, Cout0/Cin1, S0, A1, B1, Cout, S1]
        #Expected output:
        #[Cin, A1, A0, B1, B0, Cout, S1, S0]
        out.outbits = torch.cat((out.outbits[:1],
            out.outbits[offset:adder2.num_bits+offset], out.outbits[1:adder1.num_bits+1],
            out.outbits[adder2.num_bits+offset:2*adder2.num_bits+offset], out.outbits[adder1.num_bits+1:2*adder1.num_bits+1],
            out.outbits[2*adder2.num_bits+offset:], out.outbits[2*adder1.num_bits+2:3*adder1.num_bits+2]))

        return AdderRBM.fromRBM(out, adder1.num_bits+adder2.num_bits)



    """
    Creates AdderRBM type from regular RBM. The outbits of RBM must be set to the standard Adder form of
    [Cin, A1, A0, B1, B0, Cout, S1, S0] (2 bit adder).
    """
    def fromRBM(adder, num_bits):
        if not len(adder.outbits) == 3*num_bits+2:
            raise ValueError("Outbits not set properly! Should be of length 3*num_bits+2")

        out = AdderRBM(num_bits, adder.num_hidden, adder.k)
        #Reinitalizes the Adder RBM, this is mostly to ensure the outbits are correct.
        super(AdderRBM, out).__init__(adder.num_visible, adder.num_hidden,
        adder.k, learning_rate=adder.learning_rate, use_momentum=adder.use_momentum,
        momentum_coefficient=adder.momentum_coefficient, decay_type=adder.decay_type,
        weight_decay=adder.weight_decay, device=adder.device, outbits=adder.outbits)

        #copying over weights, visible bias and hidden bias
        out.weights = adder.weights.clone().detach()
        out.visible_bias = adder.visible_bias.clone().detach()
        out.hidden_bias = adder.hidden_bias.clone().detach()
        return out

class MultRBM(RBM):

    def __init__(self, num_bits, num_hidden, k, learning_rate=1e-3,  use_momentum=False, momentum_coefficient=0.5, decay_type='L2', weight_decay=1e-4, device='cpu'):
        super(MultRBM, self).__init__(4*num_bits, num_hidden, k, learning_rate=learning_rate, use_momentum=use_momentum, momentum_coefficient=momentum_coefficient, decay_type=decay_type, weight_decay=weight_decay, zeros=None, device=device, outbits=None)
        self.num_bits=num_bits

    """
    Merges multiplier with itself to double the number of bits being multiplied
    """
    def merge_mults(adder, mult):
        if not adder.num_bits == mult.num_bits:
            raise ValueError("Adder and Multiplier should have same number of bits!")
        num_bits = mult.num_bits

        M0 = copy.deepcopy(mult)
        M1 = copy.deepcopy(mult)
        M2 = copy.deepcopy(mult)
        M3 = copy.deepcopy(mult)
        FA0 = copy.deepcopy(adder)
        FA1 = copy.deepcopy(adder)
        FA2 = copy.deepcopy(adder)
        FA3 = copy.deepcopy(adder)
        FA4 = copy.deepcopy(adder)
        FA5 = copy.deepcopy(adder)

        #A0 multiplier
        mults0 = RBM.merge(M0, M1, list(zip(M0.outbits[:num_bits], M1.outbits[:num_bits])))
        #A1 multiplier
        mults1 = RBM.merge(M2, M3, list(zip(M0.outbits[:num_bits], M1.outbits[:num_bits])))

        #Merging A0 multiplier with A1 multiplier
        #B0 + B0
        temp = list(zip(mults0.outbits[num_bits:2*num_bits], mults1.outbits[num_bits:2*num_bits]))
        #B1 + B1
        temp += list(zip(mults0.outbits[4*num_bits:5*num_bits], mults1.outbits[4*num_bits:5*num_bits]))
        mults = RBM.merge(mults0, mults1, temp)


        #Merging FA0
        #A0B1[n:] + B,
        temp = list(zip(mults.outbits[6*num_bits:7*num_bits], FA0.outbits[num_bits+1:2*num_bits+1]))
        #A0B0[:n] + A
        temp += list(zip(mults.outbits[2*num_bits:3*num_bits], FA0.outbits[1:num_bits+1]))
        out_mult = RBM.merge(mults, FA0, temp)


        #Merging FA1
        #FA1 + A
        temp = list(zip(out_mult.outbits[12*num_bits+2:13*num_bits+2], FA1.outbits[1:num_bits+1]))
        #A1B0[n:] + B
        temp += list(zip(out_mult.outbits[9*num_bits:10*num_bits], FA1.outbits[num_bits+1:2*num_bits+1]))
        out_mult = RBM.merge(out_mult, FA1, temp)

        #Merging FA2
        #A0B1[:n] + A
        temp = list(zip(out_mult.outbits[5*num_bits:6*num_bits], FA2.outbits[1:num_bits+1]))
        #A1B0[:n] + B
        temp += list(zip(out_mult.outbits[8*num_bits:9*num_bits], FA2.outbits[num_bits+1:2*num_bits+1]))
        #FA0_Cout + Cin
        temp += [(out_mult.outbits[12*num_bits+1], 0)]
        out_mult = RBM.merge(out_mult, FA2, temp)


        #Merging FA3
        #FA2 + A
        temp = list(zip(out_mult.outbits[14*num_bits+5:15*num_bits+5], FA3.outbits[1:num_bits+1]))
        #A1B1[n:] + B
        temp += list(zip(out_mult.outbits[11*num_bits:12*num_bits], FA3.outbits[num_bits+1:2*num_bits+1]))
        #FA1_Cout + Cin
        temp += [(out_mult.outbits[13*num_bits+3], 0)]
        out_mult = RBM.merge(out_mult, FA3, temp)


        #Merging FA4
        #FA2_cout + A
        temp = [(15*num_bits+5, num_bits)]
        #A1B1[:n] + B
        temp += list(zip(out_mult.outbits[10*num_bits:11*num_bits], FA3.outbits[num_bits+1:2*num_bits+1]))
        #FA3_Cout + Cin
        temp += [(out_mult.outbits[14*num_bits+4], 0)]
        out_mult = RBM.merge(out_mult, FA4, temp)


        zeros = [out_mult.outbits[12*num_bits],
        out_mult.outbits[13*num_bits+2]] + list(out_mult.outbits[16*num_bits+6:17*num_bits+6])

        outbits = [out_mult.outbits[7*num_bits:8*num_bits],
        out_mult.outbits[:num_bits],
        out_mult.outbits[4*num_bits:5*num_bits],
        out_mult.outbits[num_bits:2*num_bits]]

        outbits += [out_mult.outbits[17*num_bits+6:18*num_bits+6],
        out_mult.outbits[15*num_bits+6:16*num_bits+6],
        out_mult.outbits[13*num_bits+4:14*num_bits+4],
        out_mult.outbits[3*num_bits:4*num_bits]]

        outbits = torch.cat(outbits, 0)
        zeros = torch.cat((out_mult.zeros, torch.LongTensor(zeros)), 0)
        out_mult.outbits = outbits

        out_mult.zeros = zeros

        out_mult = MultRBM.fromRBM(out_mult, 2*num_bits)
        #Should be 18*num_bits + 6 = 42!

        return out_mult



    def fromRBM(mult, num_bits):
        if not len(mult.outbits) == 4*num_bits:
            raise ValueError("Outbits not set properly! Should be of length 3*num_bits+2")

        out = MultRBM(num_bits, mult.num_hidden, mult.k)
        #Reinitalizes the Mult RBM, this is mostly to ensure the outbits are correct.
        super(MultRBM, out).__init__(mult.num_visible, mult.num_hidden, mult.k,
        learning_rate=mult.learning_rate, use_momentum=mult.use_momentum,
        momentum_coefficient=mult.momentum_coefficient, decay_type=mult.decay_type,
        weight_decay=mult.weight_decay, device=mult.device, outbits=mult.outbits, zeros=mult.zeros)

        if not mult.device == 'cpu':
            out.cuda(device=mult.device)

        #copying over weights, visible bias and hidden bias
        out.weights = mult.weights.clone().detach()

        out.visible_bias = mult.visible_bias.clone().detach()
        out.hidden_bias = mult.hidden_bias.clone().detach()

        return out

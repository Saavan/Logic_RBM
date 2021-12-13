import utils
import torch
from rbm import *

class QuantRBM(RBM):
    def __init__(self, InputRBM, maxval = 8, minval = -8, quant_n=10, quant_p=5, quant_decay = 0):
        super().__init__(InputRBM.num_visible, InputRBM.num_hidden, InputRBM.k)
        self.num_visible = InputRBM.num_visible
        self.weights = torch.zeros_like(InputRBM.weights).copy_(InputRBM.weights)
        self.visible_bias = torch.zeros_like(InputRBM.visible_bias).copy_(InputRBM.visible_bias)
        self.outbits = InputRBM.outbits
        self.zeros = InputRBM.zeros
        self.hidden_bias.copy_(InputRBM.hidden_bias)
        self.learning_rate = InputRBM.learning_rate
        self.weight_decay = InputRBM.weight_decay
        self.decay_type = InputRBM.decay_type

        self.maxval = maxval
        self.minval = minval
        self.quant_n = quant_n
        self.quant_p = quant_p
        self.quant_decay = quant_decay


    def contrastive_divergence(self, input_data, pcd=False):
        """
        Performs one iteration of contrastive divergence using the input data provided
        input data should be a torch tensor
        Contrastive divergence parameters set on RBM initialization
        This overrides the super class contrastive divergence by adding quantization
        loss terms and clamping to maximum and minimum values
        input_data - torch Tensor containing input data to train on
        pcd - True, use persistent contrastive divergence, False, use regular CD learning. False by default.
        """
        error = super().contrastive_divergence(input_data, pcd=pcd)

        #adding penalties for bit values far away from quantizations
        #Effectively an L1 weight decay towards the quantized values
        self.weights -= self.quant_decay * torch.sign(self.weights - utils.quantize_tens(self.weights, self.quant_n, self.quant_p))
        self.visible_bias -= self.quant_decay * torch.sign(self.visible_bias - utils.quantize_tens(self.visible_bias, self.quant_n, self.quant_p))
        self.hidden_bias -= self.quant_decay * torch.sign(self.hidden_bias - utils.quantize_tens(self.hidden_bias, self.quant_n, self.quant_p))

        #Clamp weights to the min anx max val specified on initialization
        self.weights = torch.clamp(self.weights, self.minval, self.maxval)
        self.visible_bias = torch.clamp(self.visible_bias, self.minval, self.maxval)
        self.hidden_bias = torch.clamp(self.hidden_bias, self.minval, self.maxval)

        return error

    def quantize(model, n_w, p_w, n_b=None, p_b=None):
        '''
        Quantizes the weights and biases of a given RBM input model and returns
        a QuantRBM Type. If n_b, p_b are not set assumed to be
        model - input model
        n_w - number of bits in weights
        p_w - point location for weights
        n_b - number of bits for biases
        p_b - point location for biases
        '''
        if n_b == None:
            n_b = n_w
        if p_b == None:
            p_b = p_w
        out = QuantRBM(model, maxval=(2**(n_w-p_w-1) - 2**(-p_w)),
                    minval=-1 * (2**(n_w-p_w-1)), quant_n=n_w, quant_p=p_w)
        
        out.to(model.device)

        out.weights = utils.quantize_tens(model.weights.clone().detach(), n_w, p_w)
        out.visible_bias = utils.quantize_tens(model.visible_bias.clone().detach(), n_b, p_b)
        out.hidden_bias = utils.quantize_tens(model.hidden_bias.clone().detach(), n_b, p_b)
        out.num_visible = model.num_visible
        out.outbits = model.outbits.clone().detach()
        out.zeros = model.zeros.clone().detach()


        return out

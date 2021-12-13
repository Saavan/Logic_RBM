import numpy as np
import torch
import torch.nn
from joblib import Parallel, delayed
import multiprocessing
import RBM.utils as utils
import pickle
import itertools
import logging
import gc

"""
This is the 

"""
class RBM():

    """
    num_visible - number of visible units
    num_hidden - number of hidden units
    k - number of cycles to use in contrastive divergence while training
    learning_rate - learning rate for training 
    use_momentum -- use momentum to aid in training
    decay_type -- either L1 or L2 decay while training
    weight_decay -- coefficient of weight decay 
    device -- cpu or gpu to house the model
    outbits -- a way to only output states that we care about, and ignore others
    zeros -- if certain states are forced to zero always, this vector tells which states
    """
    def __init__(self, num_visible, num_hidden, k, learning_rate=1e-3,
    use_momentum=False, momentum_coefficient=0.5, decay_type='L2',
    weight_decay=1e-4, device='cpu', outbits=None, zeros=None):
        self.device = device

        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k

        self.learning_rate = learning_rate
        self.decay_type = decay_type
        self.use_momentum = use_momentum
        self.momentum_coefficient = momentum_coefficient
        self.weight_decay = weight_decay

        self.sigmoid = torch.nn.Sigmoid()

        self.weights = torch.randn(num_visible, num_hidden, device=torch.device(self.device)) * 0.1
        self.visible_bias = torch.ones(num_visible, device=torch.device(self.device)) * 0.5
        self.hidden_bias = torch.zeros(num_hidden, device=torch.device(self.device))


        if use_momentum:
            self.weights_momentum = torch.zeros(num_visible, num_hidden, device=torch.device(self.device))
            self.visible_bias_momentum = torch.zeros(num_visible, device=torch.device(self.device))
            self.hidden_bias_momentum = torch.zeros(num_hidden, device=torch.device(self.device))


        self.pcd_visible = 0
        self.pcd_hidden = 0

        if outbits is None:
            self.outbits = torch.arange(self.num_visible, device=torch.device(self.device), dtype=torch.long)
        #copies over from other tensor
        elif type(outbits) == torch.Tensor:
            self.outbits = outbits.long().clone().detach()
        else:
            self.outbits = torch.LongTensor(outbits, device=torch.device(self.device))

        if zeros is None:
            self.zeros = torch.Tensor([]).long()
            self.zeros = self.zeros.to(torch.device(self.device))
        elif type(zeros) == torch.Tensor:
            self.zeros = zeros.clone().detach()
        else:
            self.zeros = torch.Tensor(zeros).long()
            self.zeros = self.zeros.to(torch.device(self.device))
    """
    Sample hidden units from visible_probabilities (or activations)
    Uses sigmoidal activation function
    visible_probabilities - torch.Tensor containing visible probabilities (or activations)
    """
    def sample_hidden(self, visible_probabilities, temperature=1):
        hidden_activations = torch.matmul(visible_probabilities, self.weights/temperature) + self.hidden_bias/temperature
        return self.sigmoid(hidden_activations)

    """
    Sample visible units from hidden_probabilities (or activations)
    Uses sigmoidal activation function
    hidden_probabilities - torch.Tensor containing hidden probabilities (or activations)
    """

    def sample_visible(self, hidden_probabilities, temperature=1):
        visible_activations = torch.matmul(hidden_probabilities, self.weights.t()/temperature) + self.visible_bias/temperature
        return self.sigmoid(visible_activations)


    """
    Performs one iteration of contrastive divergence using the input data provided
    input data should be a torch tensor
    Contrastive divergence parameters set on RBM initialization
    input_data - torch Tensor containing input data to train on
    pcd - True, use persistent contrastive divergence, False, use regular CD learning. False by default.
    """
    def contrastive_divergence(self, input_data, pcd=False):
        # Positive phase
        # <s_i s_j>_0
        positive_hidden_probabilities = self.sample_hidden(input_data)
        positive_hidden_activations = (positive_hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()
        positive_associations = torch.matmul(input_data.t(), positive_hidden_activations)

        # Negative phase
        hidden_activations = positive_hidden_activations
        if pcd and torch.is_tensor(self.pcd_visible):
            hidden_activations = (self.sample_hidden(self.pcd_visible) >= self._random_probabilities(self.num_hidden)).float()

        #Alternating gibbs sampling
        for step in range(self.k):
            visible_probabilities = self.sample_visible(hidden_activations)
            hidden_probabilities = self.sample_hidden(visible_probabilities)
            hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        negative_visible_probabilities = visible_probabilities
        negative_hidden_probabilities = hidden_probabilities

        #Saving these for later
        if pcd:
            self.pcd_visible = visible_probabilities
            self.pcd_hidden = hidden_probabilities

        # <s_i s_j>_k
        negative_associations = torch.matmul(negative_visible_probabilities.t(), negative_hidden_probabilities)

        # Update parameters

        #Use weight momentum updates
        if self.use_momentum:
            self.weights_momentum *= self.momentum_coefficient
            self.weights_momentum += (positive_associations - negative_associations)

            self.visible_bias_momentum *= self.momentum_coefficient
            self.visible_bias_momentum += torch.sum(input_data - negative_visible_probabilities, dim=0)

            self.hidden_bias_momentum *= self.momentum_coefficient
            self.hidden_bias_momentum += torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim=0)

        #Non-momentum parameters
        else:
            self.weights_momentum = positive_associations - negative_associations
            self.visible_bias_momentum = torch.sum(input_data - negative_visible_probabilities, dim = 0)
            self.hidden_bias_momentum = torch.sum(positive_hidden_probabilities - negative_hidden_probabilities, dim = 0)

        batch_size = input_data.size(0)

        self.weights += self.weights_momentum * self.learning_rate / batch_size
        self.visible_bias += self.visible_bias_momentum * self.learning_rate / batch_size
        self.hidden_bias += self.hidden_bias_momentum * self.learning_rate / batch_size


        if self.decay_type == 'L2':
            self.weights -= self.weights * self.weight_decay  #L2 weight decay
            self.visible_bias -= self.visible_bias * self.weight_decay
            self.hidden_bias -= self.hidden_bias * self.weight_decay
        elif self.decay_type == 'L1':
            self.weights -= self.weight_decay * torch.sign(self.weights) #L1 Weight decay
            self.visible_bias -= self.weight_decay * torch.sign(self.visible_bias)
            self.hidden_bias -= self.weight_decay * torch.sign(self.hidden_bias)


        # Compute reconstruction error
        error = torch.sum((input_data - negative_visible_probabilities)**2)

        return error

    """
    Trains the RBM for given number of epochs, and with the given train_loader
    train_loader - train_loader (from class torch.utils.DataLoader) to batch train on data
                    should enumerate batches as torch vectors (not tuples)
    epochs - number of epochs to train on the given training data.
    pcd - Use persistent contrastive divergence or regular
    test_fn - test function to run on RBM after every batch. Used to extract more data
    about RBM functionality
    """
    def train(self, train_loader, epochs, pcd=False, test_fn=None):
        batch_errors = []
        test_errors = []

        for epoch in range(epochs):
            epoch_error = 0.0
            for i, batch in enumerate(train_loader):
                batch = batch.view(len(batch), self.num_visible)  # flatten input data
                if not self.device == 'cpu':
                    batch = batch.to(torch.device(self.device))

                batch_error = self.contrastive_divergence(batch, pcd=pcd)
                batch_errors.append(batch_error.cpu())
                epoch_error += batch_error

            if not test_fn is None:
                acc = test_fn(self)
                test_errors.append(acc)
                logging.debug("Epoch:{0:d}, Test Accuracy:{1}, Epoch_Error:{2:f}".format(epoch, acc, epoch_error))
            else:
                logging.debug("Epoch:{0:d}, Epoch Error{1:f}".format(epoch,epoch_error))

        return batch_errors, test_errors

    """
    Generates sampling from RBM with given input_data and clamped visible features
    to clamp a feature, set the clamp bit to desired input (0 or 1), or set to -1 to not clamp
    k - number of iterations before taking a sample. defaults to 1
    input_data - data to seed the sample generation with. Units are initialized to this value before a sample is taken
    clamp - if doing partial reconstruction, what to clamp input bits to.
    TODO: This should be able to be done tensorwise! i.e. if we need a bunch of samples, we should be able to get them all at once using one matrix multiplication. This would be useful for doing testing of multiple samples.
    """
    def generate_sample(self, k=1, input_data=None, clamp=None, use_outbits=True, use_zeros=True):
        if use_outbits:
            if not clamp is None:
                temp = clamp
                clamp = torch.zeros(self.num_visible, device=torch.device(self.device)) - 1
                clamp[self.outbits] = temp

        if clamp is None:
            clamp = torch.zeros(self.num_visible, device=torch.device(self.device)) - 1
        if use_zeros:
            clamp[self.zeros] = 0

        #Input data is randomized if not present
        if input_data is None:
            input_data = self._random_probabilities(self.num_visible).round()

        #Input data should be clamped regardless of its origin
        input_data = torch.max(input_data, clamp)
        input_data = torch.min(input_data, torch.abs(clamp))


        hidden_probabilities = self.sample_hidden(input_data)
        hidden_activations = (hidden_probabilities >= self._random_probabilities(self.num_hidden)).float()

        for step in range(k):
            visible_activations = (self.sample_visible(hidden_activations) >= self._random_probabilities(self.num_visible)).float()

            #Clamps 1 values
            visible_activations = torch.max(visible_activations, clamp)

            #Clamps 0 values
            visible_activations = torch.min(visible_activations, torch.abs(clamp))

            hidden_activations = (self.sample_hidden(visible_activations) >= self._random_probabilities(self.num_hidden)).float()

        if use_outbits:
            return visible_activations[self.outbits]

        return visible_activations


    """
    Generate Statistics of this RBM via Gibbs Sampling.
    num_samples - number of samples desired
    CDk - number of contrastive divergence steps to take before getting a sample
    clamp - torch Tensor clamping values to either 0 or 1. Set values in tensor to -1 if that index
            should not be clamp
    reset_cycles - number of samples taken before resetting the markov chain (i.e. setting the input to
            random values
    num_jobs - number of parallel jobs to start to generate the statistics. Note: do not use with CUDA, as this
            could have unexpected effects
    """
    def generate_statistics(self, num_samples, CDk=1, clamp=None, reset_cycles=-1, num_jobs=1, use_outbits=True, use_zeros=True, keep_samps=False):
        if num_jobs==1:
            vals =  self._gen_stat_helper(num_samples, CDk=CDk, clamp=clamp, reset_cycles=reset_cycles, use_outbits=use_outbits, use_zeros=use_zeros, keep_samps=keep_samps)
        else:
            #Divide the number of samples up between the individual workers, so each worker gets an equal portion of samples
            #This is much faster than having each worker individually run a sample, then wait for the next one
            workers = Parallel(n_jobs=num_jobs)(delayed(self._gen_stat_helper)(int(num_samples/num_jobs), CDk=CDk, clamp=clamp, reset_cycles=reset_cycles, use_outbits=use_outbits, use_zeros=use_zeros,  keep_samps=keep_samps) for _ in range(num_jobs))
            vals = workers[0][0]
            for worker in workers[0][1:]:
                vals = utils.combine_probs(vals, worker)

        return vals

    """
    Helper function to allow easy parallelization of generate statistics
    """
    def _gen_stat_helper(self, num_samples, CDk=1, clamp=None, reset_cycles=-1, use_outbits=True, use_zeros=True, keep_samps=False):
        if use_outbits:
            bitmap = self.outbits
            temp = clamp
            clamp = torch.zeros(self.num_visible, device=torch.device(self.device)) - 1
            if not temp is None:
                clamp[self.outbits] = temp
        else:
            bitmap = torch.arange(self.num_visible, device=torch.device(self.device)).long()

        #Setting appropriate values to 0
        if use_zeros:
            clamp[self.zeros] = 0
        out = {}

        #Start point should be random probability, note the previous sample needs to be seeded with the full number
        #of visible units.
        prevSample = self._random_probabilities(self.num_visible).round()

        #Calculates powers of 2 from most signifcant bit to least, used for key creation.
        #For some reason, need to call .cuda() instead of setting device here, not sure why
        pows = torch.from_numpy(np.power(2, np.flip(np.arange(len(bitmap)), axis=0))).float()
        if not self.device == 'cpu':
            pows = pows.to(torch.device(self.device))

        if keep_samps:
            samps = [0] * num_samples
        else:
            samps = []
        for i in range(num_samples):
            sample = self.generate_sample(input_data=prevSample, k=CDk, clamp=clamp, use_outbits=False, use_zeros=use_zeros)

            #Apparently this is the most efficienty way of converting a tensor to a string?
            #Actually runs a factor of 10 faster than using the map function...
            key = utils.tensorToString(sample[bitmap])

            #Saving the order of the markov chain.
            if keep_samps:
                samps[i] = sample[bitmap]

            if key in out:
                out[key][1] += 1
            else:
                out[key] = [sample[bitmap], 1]


            prevSample = sample

            #Resets markov chain to random values occasionally
            if reset_cycles > 0 and i%reset_cycles == 0:
                prevSample = self._random_probabilities(self.num_visible).round()

        return out, samps


    """
    Generate a sample of this RBM via Gibbs Sampling. U
    num_samps - number of parallel samples desired
    k - number of contrastive divergence steps to take before getting a sample
    clamp - torch Tensor clamping values to either 0 or 1. Set values in tensor to -1 if that index
            should not be clamp
    input_visible - input visible data to start sampling from
    use_outbits -- True for outputting only the outbits of self, False for outputting all intermediate states
    use_zeros -- True for zero clamping with the self.zeros, False for no zero clamping
    temperature -- scaling factor for sampling at different temperatures
    """
    def tensgenerate_sample(self, num_samps, k=1, input_visible=None, clamp=None, use_outbits=True, use_zeros=True, temperature=1):
        if use_outbits:
            if not clamp is None:
                temp = clamp
                clamp = torch.zeros(self.num_visible, device=torch.device(self.device)) - 1
                clamp[self.outbits] = temp

        if clamp is None:
            clamp = torch.zeros(self.num_visible, device=torch.device(self.device)) - 1

        if use_zeros:
            if len(clamp.size()) > 1:
                clamp[:, self.zeros] = 0
            else:
                clamp[self.zeros] = 0

         #Input data is randomized if not present
        if input_visible is None:
            input_visible = torch.rand(num_samps, self.num_visible, device=torch.device(self.device)).round().float()

         #Input data should be clamped regardless of its origin
        input_visible = torch.max(input_visible, clamp)
        input_visible = torch.min(input_visible, torch.abs(clamp))

        visible_activations = input_visible

        for step in range(k):
            hidden_activations = (self.sample_hidden(visible_activations, temperature=temperature) >= torch.rand(num_samps, self.num_hidden, device=torch.device(self.device))).float()


            visible_activations = (self.sample_visible(hidden_activations, temperature=temperature) >= torch.rand(num_samps, self.num_visible, device=torch.device(self.device))).float()

             #Clamps 1 values
            visible_activations = torch.max(visible_activations, clamp)

             #Clamps 0 values
            visible_activations = torch.min(visible_activations, torch.abs(clamp))

        if use_outbits:
            return visible_activations[:, self.outbits]

        return visible_activations

    """
    Generate full tensor statistics of this RBM via Gibbs Sampling. U
    num_samps - number of sequential samples to take
    num_chains -- number of parallel chains to operate at once
    clamp - torch Tensor clamping values to either 0 or 1. Set values in tensor to -1 if that index
            should not be clamp
    input_data - input visible data to start sampling from
    use_outbits -- True for outputting only the outbits of self, False for outputting all intermediate states
    use_zeros -- True for zero clamping with the self.zeros, False for no zero clamping
    temperature -- scaling factor for sampling at different temperatures
    keep_samps -- True, returns the full sampling run, False only returns the dictionary of samples
    Stride -- number of sampling steps to perform before collapsing samples into a counts dictionary
    Returns a dictionary of samples, where the key is the sample, and the value is the count of occurences of that sample
    """
    def tensgenerate_statistics(self, num_samples, num_chains, k=1, clamp=None,
                                use_outbits=True, use_zeros=True, input_data=None,
                                keep_samps=False, temperature=1, stride=16384):
        if use_outbits:
            bitmap = self.outbits
            temp = clamp
            clamp = torch.zeros(num_chains, self.num_visible, device=torch.device(self.device)) - 1
            if not temp is None:
                clamp[:, self.outbits] = temp
        else:
            bitmap = torch.arange(self.num_visible, device=torch.device(self.device)).long()
        if use_zeros:
            if len(clamp.size()) > 1:
                clamp[:, self.zeros] = 0
            else:
                clamp[self.zeros] = 0
        out = [{} for _ in range(num_chains)]
         #Start point should be random probability, note the previous sample needs to be seeded with the      full number
         #of visible units.
        if input_data is None:
            sample = torch.randint(0, 2, (num_chains, self.num_visible),
                                dtype=torch.float32, device=torch.device(self.device))
        else:
            sample = input_data

        samps = []

        for ind in np.arange(0, num_samples, stride):
            #Do up to the stride length samples before collecting
            maxind = stride if ind + stride < num_samples else num_samples - ind

            out, sample, tempSamps = self.__tensHelper(maxind, num_chains, out, bitmap, clamp=clamp, input_data=sample, k=1, use_zeros=use_zeros,
                                        keep_samps=keep_samps, temperature=temperature)

            if ind == 0 and keep_samps:
                samps = tempSamps
            elif keep_samps:
                samps = torch.cat((samps, tempSamps), dim=0)

            # del(tempSamps)
            #Collect garbage
            gc.collect()
            #Reset the cache


        return out, samps

    """
    Performs inner loop for tensgenerate_statistics. This makes sure things get properly garbage colleted
    """
    def __tensHelper(self, maxind, num_chains, out, bitmap, clamp=None, input_data=None, k=1, use_zeros=True,
                                keep_samps=False, temperature=1):
        tempSamps = torch.empty((maxind, num_chains,
                            bitmap.size(0)), dtype=torch.float32, device=torch.device(self.device))

        if clamp is None:
            clamp = torch.zeros(num_chains, self.num_visible, dtype=torch.float32, device=torch.device(self.device)) - 1
        if input_data is None:
            sample = torch.randint(0, 2, (num_chains, self.num_visible),
                                    dtype=torch.float32, device=torch.device(self.device))
        else:
            sample = input_data
        for i in range(maxind):
            sample = self.tensgenerate_sample(num_chains, input_visible=sample,
                                k=k, clamp=clamp, use_outbits=False,
                                use_zeros=use_zeros, temperature=temperature)

            #Saving some of the samples so we can accumulate them later
            tempSamps[i] = sample[:, bitmap]


        #To assemble all of the samples and count them, move back to CPU
        if not self.device == 'cpu':
            tempSamps = tempSamps.cpu()

        for samp_ind in range(maxind):
            for key_ind, val in enumerate(tempSamps[samp_ind]):
                #Gets the raw bytes and converts them to a string to use as key
                #This is significantly faster than any other method.
                #This operation doesn't work on the GPU which is why it
                #needs to be done on the CPU
                key = val.byte().numpy().tostring()

                if key in out[key_ind]:
                    #out[key_ind][key][1] += 1
                    out[key_ind][key] += 1
                else:
                    #out[key_ind][key] = [val, 1]
                    out[key_ind][key] = 1
        if keep_samps:
            return out, sample, tempSamps

        del(tempSamps)
        return out, sample, None

        """
        Merges two RBMs using given nodes to merge
        Functions by combining the weight and bias matrices, with 0s along all non-merged nodes
        Takes all learning parameters from first RBM

        first - first RBM to be merged into. all parameters for new RBM taken from this RBM
        second - second RBM to merge weights from. Only parameters taken from this RBM is weight and bias matrices
        nodes - a list of tuples, referring to nodes to be merged, the first value being the node from first RBM, the second value being the node from the second RBM to be merged with. The index of the merged node is the index of the first RBM to be merged (the first value in the tuple)
        return - a new RBM that merges the weight matrix of the first RBM and the second RBM
        """
    def merge(first, second, nodes):
        #Merged RBM uses minimal parameters for construction, the rest are taken from first RBM
        merged = RBM(first.num_visible + second.num_visible - len(nodes), first.num_hidden + second.num_hidden, first.k,
            use_momentum=first.use_momentum, momentum_coefficient=first.momentum_coefficient, decay_type=first.decay_type,
            weight_decay=first.weight_decay, device=first.device, outbits=first.outbits)

        names = [attr for attr in dir(first) if not callable(getattr(first, attr)) and not attr.startswith("__")]
        for attr in dir(merged):
            if attr in names:
                setattr(merged, attr, getattr(merged, attr))

        merged.weights = torch.zeros(first.num_visible + second.num_visible - len(nodes), first.num_hidden + second.num_hidden)

        #Putting in weights from first RBM
        merged.weights[0:first.num_visible, 0:first.num_hidden] = first.weights

        #Hidden nodes are just concatenated together
        merged.hidden_bias = torch.cat((first.hidden_bias, second.hidden_bias))

        #Initializing visible bias for first node
        merged.visible_bias[0:first.num_visible] = first.visible_bias

        #Initializing outbits
        merged.outbits = torch.empty(first.outbits.size(0) + second.outbits.size(0), dtype=torch.long)
        merged.outbits[0:first.outbits.size(0)] = first.outbits
        merged.outbits[first.outbits.size(0):] = second.outbits + first.num_visible

        #Initializing zeros
        merged.zeros = torch.empty(first.zeros.size(0) + second.zeros.size(0), dtype=torch.long)
        merged.zeros[0:first.zeros.size(0)] = first.zeros
        merged.zeros[first.zeros.size(0):] = second.zeros + first.num_visible

        switched = []

        #Merged nodes being combined via putting together weights and biases
        for index in nodes:
            #Weights are concatenated together
            merged.weights[index[0], first.num_hidden:] = second.weights[index[1], :]

            #Bias for merged node is sum of both biases
            merged.visible_bias[index[0]] += second.visible_bias[index[1]]

            #Need to modify outbits when performing merge operation
            if index[1]+first.num_visible in merged.outbits:
                #If both indices are in the merged outbits, then we get rid of the second one
                if index[0] in merged.outbits:
                    merged.outbits = torch.masked_select(merged.outbits, ~torch.eq(merged.outbits, first.num_visible + index[1]))
                #If the first index is not in the outbits, replace all instances of the second index with the first one
                else:
                    merged.outbits.index_fill_(0, torch.nonzero(torch.eq(merged.outbits, first.num_visible+index[1])).flatten(), index[0])

            #Keep track of indices that have been switched
            switched.append(index[1])

        #Outbits greater than the one removed get scaled down by one
        acc_out = torch.zeros_like(merged.outbits)
        acc_zeros = torch.zeros_like(merged.zeros)
        for ind in switched:
            acc_out = acc_out + torch.gt(merged.outbits, ind + first.num_visible).long()
            acc_zeros = acc_zeros + torch.gt(merged.zeros, ind + first.num_visible).long()
        merged.outbits = merged.outbits - acc_out
        merged.zeros = merged.zeros - acc_zeros

        ind = 0
        for x in range(second.num_visible):
            #All non-merged nodes are put into weight matrix
            if x not in switched:
                #second weights take up bottom right of weight matrix
                merged.weights[first.num_visible + ind, first.num_hidden:] = second.weights[x, :]
                merged.visible_bias[first.num_visible + ind] = second.visible_bias[x]
                ind += 1

        return merged

    """
    collapses nodes in the model together. This operation is similar to merging, except that it is a merge with itself
    Has the same mathematical properties as merging.
    The first index retains the same properties as the two merged together
    nodes - list of nodes to in self to merged together, should be a list of lists with each individual list being two nodes in self
    """
    def collapse(self, nodes):
        removed = []
        for node in nodes:
            #making sure the indices are still correct after mergers have been done
            ind0 = node[0]
            ind1 = node[1]
            min0 = 0
            min1 = 0
            for ind in removed:
                if ind < ind1:
                    min1 += 1
                if ind < ind0:
                    min0 += 1
            ind0 -= min0
            ind1 -= min1
            #Weights and visible biases should be added together if they have anything in common
            self.weights[ind0, :] += self.weights[ind1, :]
            self.visible_bias[ind0] += self.visible_bias[ind1]
            #Removing the second index from the weights & visible bias matrix
            self.weights = torch.cat((self.weights[:ind1, :], self.weights[ind1+1:, :]), dim=0)
            self.visible_bias = torch.cat((self.visible_bias[:ind1], self.visible_bias[ind1+1:]), dim=0)
            #If its in outbits or zeros, remove it
            self.num_visible -= 1
            removed.append(node[1])
        print(removed)

        for ind1 in removed:
            if ind1 in self.outbits:
                print('removing index:', ind1)
                self.outbits = torch.masked_select(self.outbits, ~torch.eq(self.outbits, ind1))
                self.zeros = torch.masked_select(self.zeros, ~torch.eq(self.zeros, ind1))
        acc_out = torch.zeros_like(self.outbits)
        acc_zeros = torch.zeros_like(self.zeros)
        #Making sure each of the outbits get scaled down by the correct factor
        for ind in removed:
            acc_out = acc_out + torch.gt(self.outbits, ind).long()
            acc_zeros = acc_zeros + torch.gt(self.zeros, ind).long()
        print('outbits before collapse', self.outbits, 'len:', len(self.outbits))
        self.outbits = self.outbits - acc_out
        print('outbits after collapse', self.outbits, 'len:', len(self.outbits))
        self.zeros = self.zeros - acc_zeros
        return



    """
    Saving an RBM in the given filename. Does not save full RBM, only non-private variables and functions
    filename - The filename to be saved in. Should be a string represnting a full path to the file and extension. Standard
    exension is ".p"
    """
    def save(self, filename):
        #Saving only non-private variables. This is so that if methods are changed in future RBMs, older version
        #RBMs should still be able to be loaded.
        names = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        attrs = [getattr(self, attr) for attr in names]
        statedict = dict(zip(names, attrs))
        pickle.dump(statedict, open(filename, 'wb'))
        return

    def populate_model(a, lambd, data):
        """Populates model based to create an RBM which exactly models the data distribution given
        """
        model = RBM(len(data[0]), len(data), 1)
        for i,v in enumerate(data):
            w = a * (v - 0.5)
            c = -1 * torch.dot(w, v) + lambd
            model.weights[:, i] = w
            model.hidden_bias[i] = c
        model.visible_bias = torch.zeros(model.num_visible)
        return model

    """
    load the RBM from the given filename. filename should be a path to the pickled RBM
    """
    def load(filename):
        out = RBM(1, 1, 1)
        loaded = pickle.load(open(filename, 'rb'))
        #If statement because some RBMs had their entire class pickled rather than just a state dict
        #Support for RBMs that were fully pickled rather than using a statedict
        if type(loaded) == type(RBM(1, 1, 1)):
            out = RBM(loaded.num_visible, loaded.num_hidden, loaded.k,
            learning_rate=loaded.learning_rate,  use_momentum=loaded.use_momentum,
            momentum_coefficient=loaded.momentum_coefficient,
            decay_type=loaded.decay_type, weight_decay=loaded.weight_decay, device=loaded.device)
            names = [attr for attr in dir(loaded) if not callable(getattr(loaded, attr)) and not attr.startswith("__")]
            attrs = [getattr(loaded, attr) for attr in names]
            for attr in dir(out):
                if attr in names:
                    setattr(out, attr, getattr(loaded, attr))

        elif type(loaded) == type({}):
            out = RBM(loaded['num_visible'], loaded['num_hidden'], loaded['k'],
            learning_rate=loaded['learning_rate'],  use_momentum=loaded['use_momentum'],
            momentum_coefficient=loaded['momentum_coefficient'], decay_type=loaded['decay_type'],
            weight_decay=loaded['weight_decay'], device=loaded['device'])
            for attr in dir(out):
                if attr in loaded:
                    setattr(out, attr, loaded[attr])

        else:
            raise TypeError("Tried to load an invalid file!")

        return out

    """
    Moves RBM to GPU for CUDA use. Does this by getting all attributes callin attr = attr.cuda()
    Default is to move cuda:1 which is the second graphics card
    """
    def cuda(self, device='cuda:1'):
        self.device = device
        for attr in dir(self):
            if hasattr(getattr(self, attr), "cuda") and not attr == "__class__":
                #Sets attr = attr.cuda()
                setattr(self, attr, getattr(self, attr).to(device=torch.device(self.device)))

    """
    Moves RBM to GPU for CUDA use. Does this by getting all attributes calling attr = attr.cpu()
    """
    def cpu(self):
        self.device = 'cpu'
        for attr in dir(self):
            if hasattr(getattr(self, attr), "cpu") and not attr == "__class__":
                #Sets attr = attr.cpu()
                setattr(self, attr, getattr(self, attr).cpu())

    def to(self, device):
        ''' Moves model to the device specified (either cpu or cuda:0 or cuda:1)'''
        self.device = device
        for attr in dir(self):
            if hasattr(getattr(self, attr), "cuda") and not attr == "__class__":
                #Sets attr = attr.cuda()
                setattr(self, attr, getattr(self, attr).to(device=torch.device(self.device)))

    def _random_probabilities(self, num):
        random_probabilities = torch.rand(num, device=torch.device(self.device))

        return random_probabilities

    """
    Calculates the free energy of this state given a set of visible and hidden states
    Does this in a matrix wide way, so multiple visible and hidden states can be passed in
    visible_states - states to set the visible units into, should be torch floatTensor
    hidden-states - states to set the hidden units to, should be a torch floatTensor
    Note: visible_states and hidden_states are expected to have dimensions num_visible x 2^num_visible and num_hidden x 2^num_hidden
    """
    def free_energy(self, visible_states, hidden_states):
        if len(visible_states.size()) == 1:
            visible_states = torch.unsqueeze(visible_states, 0)
        if len(hidden_states.size()) == 1:
            hidden_states = torch.unsqueeze(hidden_states, 0)
        vis_bias = torch.unsqueeze(torch.matmul(self.visible_bias, visible_states), 1)
        hid_bias = torch.matmul(self.hidden_bias, hidden_states)
        energy = torch.matmul(torch.matmul(visible_states.t(), self.weights), hidden_states)
        return (-1 * (vis_bias + hid_bias + energy))


    """
    Calculates the probability of a visible state occuring. Does this by marginalizing over all hidden states.
    If partition==True, explicitly calculates the partition, otherwise just calculates the Free Energy
    v - states to set the visible units to, should be torch floatTensor
    """
    def prob(self, v, partition=False, log=True):
        if partition:
            part = self.partition(log=True)
        else:
            part = torch.tensor([0.], device=self.device, dtype=torch.float)
        if len(v.size()) == 1:
            v = torch.unsqueeze(v, 0)
        w_v = torch.matmul(v, self.weights)

        bias = torch.matmul(v, self.visible_bias)

        hid = torch.exp(w_v + self.hidden_bias)

        x = torch.sum(torch.log1p(hid), 1)

        ans = x + bias - part

        if log:
            return ans
        else:
            return torch.exp(ans)


    """
    Calculates the partition function for this RBM model
    Generally very slow for a large number of states, do not use for num_visible > 20
    """
    def partition(self, stride=16, log=False):
        part = self.prob(torch.zeros(self.num_visible), log=True)
        for ind in np.arange(0, self.num_visible, stride):
            states = utils.intToTens(torch.arange(2**int(ind), 2**(min(int(ind+stride), self.num_visible))).float())
            if self.num_visible > states.size(1):
                states = torch.cat((torch.zeros(states.size(0), self.num_visible - states.size(1)), states), 1)
            vis = self.prob(states, partition=False, log = True)
            part = torch.unsqueeze(torch.logsumexp(torch.cat((part, vis), 0), 0), 0)

        if log:
            return part
        else:
            return torch.exp(part)


    """
    Returns a vector of all visible state probabilities
    Generally very slow for a large number of units, do not use for num_units > 20
    """
    def probs(self, stride=16, temperature = 1):
        vis = self.prob(torch.zeros(self.num_visible, device = self.device), log=True)
        for ind in np.arange(0, self.num_visible, stride):
            states = utils.intToTens(torch.arange(2**int(ind), 2**(min(int(ind+stride), self.num_visible)), device = self.device).float())
            if self.num_visible > states.size(1):
                states = torch.cat((torch.zeros(states.size(0), self.num_visible - states.size(1), device = self.device), states), 1)
            vis = torch.cat((vis, self.prob(states, partition=False, log = True)), 0)

        vis = vis/temperature - torch.logsumexp(vis/temperature, 0)
        return torch.exp(vis)

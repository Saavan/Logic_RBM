import os
import os.path
import sys
import torch
import numpy as np
import logging
import yaml
import signal


from torch.utils.data import Dataset, DataLoader

import rbm_datasets
import time
import tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LogicRBM.rbm import *
import RBM.utils as utils
from RBM.quant_rbm import QuantRBM

#Ignore SIGHUP, so when session closed remotely, there are no problems
signal.signal(signal.SIGHUP, signal.SIG_IGN)

########## CONFIGURATION ##########
param_file = open(sys.argv[1], 'r')
params = yaml.load(param_file)


model = QuantRBM(RBM.load('trained/' + params['model_name'] + '.p'),
                maxval=(2**(params['quant']['bits']-params['quant']['point']-1) - 2**(-params['quant']['point'])),
                minval=(-1 * 2**(params['quant']['bits']-params['quant']['point']-1) - 2**(-params['quant']['point'])),
                quant_n = params['quant']['bits'], quant_p = params['quant']['point'], quant_decay = 0)

if params['type'] == 'Mult':
    datatype = rbm_datasets.MultDataset
elif params['type'] == 'Adder':
    datatype = rbm_datasets.AdderDataset
elif params['type'] == 'MergeMult':
    datatype = rbm_datasets.MergeMultDataset
else:
    raise ValueError('{0} is not a valid type!'.format(params['type']))


dataset = datatype(params['NUM_BITS'], params['dataset']['size'],
    random=params['dataset']['random'], prealloc=params['dataset']['prealloc'],
    shuffle=params['dataset']['shuffle'], block_size=params['dataset']['block_size'])

log_dir = "logs/"
file_name = "{0}{1}_{2}b{3}p_{4}_0".format(params['type'], params['NUM_BITS'],
    params['quant']['bits'], params['quant']['point'],
    time.strftime("%y%m%d", time.localtime()))
log_ext = ".log"

log_name = log_dir + file_name + log_ext

i = 0
#Makes sure that each training session gets its on file names
while os.path.isfile(log_name):
    i += 1
    file_name = file_name[:-1] + str(i)
    log_name = log_dir + file_name + log_ext

logging.basicConfig(level=logging.DEBUG, filename=log_name, filemode="a+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


if params['type'] == 'Mult' or params['type'] == 'MergeMult':
    modtype = MultRBM
    test1 = tests.test_mult
    test2 = tests.test_div
    test3 = tests.test_fact
elif params['type'] == 'Adder':
    modtype = AdderRBM
    test1 = tests.test_add
    test2 = tests.test_sub
    test3 = tests.test_rev

#These are parameters for the max value trainings
max_train = params['max_train']

#Test function to be performed after each sub_epoch of training
#generally this is a shorter/quicker test than the longer super_epoch
def test_max(model):
    qmodel = modtype.fromRBM(model, params['NUM_BITS'])
    workers1 = test1(qmodel, max_train['NUM_CASES'], 1, max_train['NUM_SAMPLES'])
    workers2 = test2(qmodel, max_train['NUM_CASES'], 1, max_train['NUM_SAMPLES'])
    workers3 = test3(qmodel, max_train['NUM_CASES'], 1, max_train['NUM_SAMPLES'])

    out = [workers1/max_train['NUM_CASES'], workers2/max_train['NUM_CASES'], workers3/max_train['NUM_CASES']]
    if params['type'] == 'Mult' or params['type'] == 'MergeMult':
        logging.info("[Mult, Div, Fact] num_cases={0}, num_samples={1}".format(max_train['NUM_CASES'], max_train['NUM_SAMPLES']))
    else:
        logging.info("[Add, Sub, Rev] num_cases={0}, num_samples={1}".format(max_train['NUM_CASES'], max_train['NUM_SAMPLES']))
    logging.info(str(out))
    return out

#These are parameters for the quantization training
quant_train = params['quant_train']

#Test function to be performed after each sub_epoch of training
#generally this is a shorter/quicker test than the longer super_epoch
def test_quant(model):
    qmodel = modtype.fromRBM(QuantRBM.quantize(model, params['quant']['bits'], params['quant']['point']),
                    params['NUM_BITS'])
    workers1 = test1(qmodel, quant_train['NUM_CASES'], 1, quant_train['NUM_SAMPLES'])
    workers2 = test2(qmodel, quant_train['NUM_CASES'], 1, quant_train['NUM_SAMPLES'])
    workers3 = test3(qmodel, quant_train['NUM_CASES'], 1, quant_train['NUM_SAMPLES'])

    out = [workers1/quant_train['NUM_CASES'], workers2/quant_train['NUM_CASES'], workers3/quant_train['NUM_CASES']]
    if params['type'] == 'Mult' or params['type'] == 'MergeMult':
        logging.info("[Mult, Div, Fact] num_cases={0}, num_samples={1}".format(quant_train['NUM_CASES'], quant_train['NUM_SAMPLES']))
    else:
        logging.info("[Add, Sub, Rev] num_cases={0}, num_samples={1}".format(quant_train['NUM_CASES'], quant_train['NUM_SAMPLES']))
    logging.info(str(out))
    return out

########## Retraining model for maximum value constraint ##########
logging.info('Training {0} bit {1}...'.format(params['NUM_BITS'], params['type']))
logging.info('Parameters:')
for val in params:
    logging.info('{0}:{1}'.format(val, params[val]))


#initialize rbm and train loader for given hyper parameters
train_loader = torch.utils.data.DataLoader(dataset,
    batch_size=max_train['BATCH_SIZE'], shuffle=False, pin_memory=params['CUDA'])


if params['CUDA']:
    model.cuda(device=params['DEVICE'])

t0 = time.time()
tprev = time.time()

model.learning_rate = max_train['LEARN_RATE']
model.k = max_train['CD_K']
model.weight_decay = max_train['WEIGHT_DECAY']


logging.info('Baseline (goal) performance after Super Epoch')
if params['type'] == 'Mult' or params['type'] == 'MergeMult':
    logging.info("[Mult, Div, Fact] num_cases={0}, num_samples={1}"
        .format(max_train['NUM_CASES_SUPER'], max_train['NUM_SAMPLES_SUPER']))
else:
    logging.info("[Add, Sub, Rev] num_cases={0}, num_samples={1}"
        .format(max_train['NUM_CASES_SUPER'], max_train['NUM_SAMPLES_SUPER']))
qmodel = modtype.fromRBM(model, params['NUM_BITS'])
workers1 = test1(qmodel, max_train['NUM_CASES_SUPER'], 1,  max_train['NUM_SAMPLES_SUPER'])
workers2 = test2(qmodel, max_train['NUM_CASES_SUPER'], 1,  max_train['NUM_SAMPLES_SUPER'])
workers3 = test3(qmodel, max_train['NUM_CASES_SUPER'], 1,  max_train['NUM_SAMPLES_SUPER'])
logging.info("[{0}, {1}, {2}]".format(workers1/max_train['NUM_CASES_SUPER'],
            workers2/max_train['NUM_CASES_SUPER'], workers3/max_train['NUM_CASES_SUPER']))



logging.info('Baseline (goal) performance after Sub Epoch')
if params['type'] == 'Mult' or params['type'] == 'MergeMult':
    logging.info("[Mult, Div, Fact] num_cases={0}, num_samples={1}"
        .format(max_train['NUM_CASES'], max_train['NUM_SAMPLES']))
else:
    logging.info("[Add, Sub, Rev] num_cases={0}, num_samples={1}"
        .format(max_train['NUM_CASES'], max_train['NUM_SAMPLES']))
workers1 = test1(qmodel, max_train['NUM_CASES'], 1,  max_train['NUM_SAMPLES'])
workers2 = test2(qmodel, max_train['NUM_CASES'], 1,  max_train['NUM_SAMPLES'])
workers3 = test3(qmodel, max_train['NUM_CASES'], 1,  max_train['NUM_SAMPLES'])
logging.info("[{0}, {1}, {2}]".format(workers1/max_train['NUM_CASES'],
            workers2/max_train['NUM_CASES'], workers3/max_train['NUM_CASES']))


try:
    for x in range(max_train['SUPER_EPOCHS']):
        logging.info("Started at: " + time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime()))
        logging.info('Learning Rate:{0}, Weight_Decay:{1}, k={2}'
            .format(model.learning_rate, model.weight_decay, model.k))
        err = model.train(train_loader, max_train['SUB_EPOCHS'], test_fn=test_max, pcd=params['PCD'])

        tnow = time.time()
        logging.info('time for this iteration=' + str(tnow - tprev))
        tprev = time.time()


        logging.info("SUPER EPOCH FINISHED, CHANGING MODEL PARAMETERS, PERFORMING FULL TEST")
        model.k = int(model.k * max_train['CDK_GROWTH_RATE'])
        model.learning_rate = model.learning_rate/max_train['LEARNING_DECAY']
        model.weight_decay = model.weight_decay/max_train['LEARNING_DECAY']

        if params['type'] == 'Mult' or params['type'] == 'MergeMult':
            logging.info("[Mult, Div, Fact] num_cases={0}, num_samples={1}"
                .format(max_train['NUM_CASES_SUPER'], max_train['NUM_SAMPLES_SUPER']))
        else:
            logging.info("[Add, Sub, Rev] num_cases={0}, num_samples={1}"
                .format(max_train['NUM_CASES_SUPER'], max_train['NUM_SAMPLES_SUPER']))

        #testing after each super epoch is generally more intensive than sub epoch
        qmodel = modtype.fromRBM(model, params['NUM_BITS'])
        workers1 = test1(qmodel, max_train['NUM_CASES_SUPER'], 1,  max_train['NUM_SAMPLES_SUPER'])
        workers2 = test2(qmodel, max_train['NUM_CASES_SUPER'], 1,  max_train['NUM_SAMPLES_SUPER'])
        workers3 = test3(qmodel, max_train['NUM_CASES_SUPER'], 1,  max_train['NUM_SAMPLES_SUPER'])
        logging.info("[{0}, {1}, {2}]".format(workers1/max_train['NUM_CASES_SUPER'],
            workers2/max_train['NUM_CASES_SUPER'], workers3/max_train['NUM_CASES_SUPER']))
except Exception as e:
    logging.fatal(e, exc_info=True)
    logging.info('TRAINING FAILED, MODEL SAVING AND EXITING')



t1 = time.time()
logging.info('total time taken=' + str(t1 - t0))
logging.info("Finished at: " + time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime()))



########## Retraining model for quantization constraint ##########
logging.info('Starting Quantization Re-training')

#initialize rbm and train loader for given hyper parameters
train_loader = torch.utils.data.DataLoader(dataset,
    batch_size=quant_train['BATCH_SIZE'], shuffle=False, pin_memory=params['CUDA'])


if params['CUDA']:
    model.cuda(device=params['DEVICE'])

t0 = time.time()
tprev = time.time()
#Main training loop over super epoch

model.learning_rate = quant_train['LEARN_RATE']
model.k = quant_train['CD_K']
model.weight_decay = quant_train['WEIGHT_DECAY']
model.quant_decay = quant_train['QUANT_DECAY']

try:
    for x in range(quant_train['SUPER_EPOCHS']):
        logging.info("Started at: " + time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime()))
        logging.info('Learning Rate:{0}, Weight_Decay:{1}, k={2}, quant_decay={3}'
            .format(model.learning_rate, model.weight_decay, model.k, model.quant_decay))
        err = model.train(train_loader, quant_train['SUB_EPOCHS'], test_fn=test_quant, pcd=params['PCD'])

        tnow = time.time()
        logging.info('time for this iteration=' + str(tnow - tprev))
        tprev = time.time()


        logging.info("SUPER EPOCH FINISHED, CHANGING MODEL PARAMETERS, PERFORMING FULL TEST")
        model.k = int(model.k * quant_train['CDK_GROWTH_RATE'])
        model.quant_decay = model.quant_decay * quant_train['QUANT_GROWTH_RATE']
        model.learning_rate = model.learning_rate/quant_train['LEARNING_DECAY']
        model.weight_decay = model.weight_decay/quant_train['LEARNING_DECAY']

        if params['type'] == 'Mult' or params['type'] == 'MergeMult':
            logging.info("[Mult, Div, Fact] num_cases={0}, num_samples={1}"
                .format(quant_train['NUM_CASES_SUPER'], quant_train['NUM_SAMPLES_SUPER']))
        else:
            logging.info("[Add, Sub, Rev] num_cases={0}, num_samples={1}"
                .format(quant_train['NUM_CASES_SUPER'], quant_train['NUM_SAMPLES_SUPER']))

        #testing after each super epoch is generally more intensive than sub epoch
        qmodel = modtype.fromRBM(QuantRBM.quantize(model, params['quant']['bits'], params['quant']['point']),
                        params['NUM_BITS'])
        workers1 = test1(qmodel, quant_train['NUM_CASES_SUPER'], 1,  quant_train['NUM_SAMPLES_SUPER'])
        workers2 = test2(qmodel, quant_train['NUM_CASES_SUPER'], 1,  quant_train['NUM_SAMPLES_SUPER'])
        workers3 = test3(qmodel, quant_train['NUM_CASES_SUPER'], 1,  quant_train['NUM_SAMPLES_SUPER'])
        logging.info("[{0}, {1}, {2}]".format(workers1/quant_train['NUM_CASES_SUPER'],
            workers2/quant_train['NUM_CASES_SUPER'], workers3/quant_train['NUM_CASES_SUPER']))
except Exception as e:
    logging.fatal(e, exc_info=True)
    logging.info('TRAINING FAILED, MODEL SAVING AND EXITING')



t1 = time.time()
logging.info('total time taken=' + str(t1 - t0))
logging.info("Finished at: " + time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime()))



########## Save it! #################
model.cpu()
outname = 'trained/' + file_name + '.p'
RBM.save(model, outname)
logging.info("Saved to: " + outname)

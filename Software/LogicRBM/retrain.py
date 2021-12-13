import os
import os.path
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import rbm_datasets
import utils
import time
import pickle
import copy
import cProfile
import tests
import logging
import yaml
import signal

from rbm import *

#Ignore SIGHUP, so when session closed remotely, there are no problems
signal.signal(signal.SIGHUP, signal.SIG_IGN)

########## CONFIGURATION ##########
param_file = open(sys.argv[1], 'r')
params = yaml.load(param_file)



add_file = 'trained/' + params['adder_name'] + '.p'
mult_file = 'trained/' + params['mult_name'] + '.p'

if params['type'] == 'Mult':
    datatype = rbm_datasets.MergeMultDataset
else:
    raise ValueError('{0} is not a valid type!'.format(params['type']))


dataset = datatype(params['NUM_BITS'], params['dataset']['size'],
    random=params['dataset']['random'], prealloc=params['dataset']['prealloc'],
    shuffle=params['dataset']['shuffle'], block_size=params['dataset']['block_size'])


log_dir = "logs/"
file_name = "{0}BitMerge{1}_{2}_0".format(params['NUM_BITS'], params['type'],
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




if params['type'] == 'Mult':
    test1 = tests.test_mult
    test2 = tests.test_div
    test3 = tests.test_fact

#Test function to be performed after each sub_epoch of training
#generally this is a shorter/quicker test than the longer super_epoch
def test(model):
    workers1 = test1(model, params['NUM_CASES'], 1, params['NUM_SAMPLES'])
    workers2 = test2(model, params['NUM_CASES'], 1, params['NUM_SAMPLES'])
    workers3 = test3(model, params['NUM_CASES'], 1, params['NUM_SAMPLES'])

    out = [workers1/params['NUM_CASES'], workers2/params['NUM_CASES'], workers3/params['NUM_CASES']]
    if params['type'] == 'Mult':
        logging.info("[Mult, Div, Fact] num_cases={0}, num_samples={1}".format(params['NUM_CASES'], params['NUM_SAMPLES']))
    else:
        logging.info("[Add, Sub, Rev] num_cases={0}, num_samples={1}".format(params['NUM_CASES'], params['NUM_SAMPLES']))
    logging.info(str(out))
    return out


########## TRAINING MODEL ##########
logging.info('Training {0} bit {1}...'.format(params['NUM_BITS'], params['type']))
logging.info('Parameters:')
for val in params:
    logging.info('{0}:{1}'.format(val, params[val]))


arr = []
fa_errors = []
epoch_errors = []
test_errors = []


#initialize rbm and train loader for given hyper parameters
train_loader = torch.utils.data.DataLoader(dataset,
    batch_size=params['BATCH_SIZE'], shuffle=False, pin_memory=params['CUDA'])

if params['type'] == 'Mult':
    modtype = MultRBM

adder = AdderRBM.fromRBM(RBM.load(add_file), int(params['NUM_BITS']/2))
mult = MultRBM.fromRBM(RBM.load(mult_file), int(params['NUM_BITS']/2))

model = MultRBM.merge_mults(adder, mult)
model.learning_rate = params['LEARN_RATE']
model.k = params['CD_K']
model.weight_decay = params['WEIGHT_DECAY']


if params['CUDA']:
    model.cuda(device=params['DEVICE'])

#testing after each super epoch is generally more intensive than sub epoch
logging.info("Starting Errors")
logging.info("[Mult, Div, Fact] num_cases={0}, num_samples={1}"
                .format(params['NUM_CASES_SUPER'], params['NUM_SAMPLES_SUPER']))
workers1 = test1(model, params['NUM_CASES_SUPER'], 1,  params['NUM_SAMPLES_SUPER'])
workers2 = test2(model, params['NUM_CASES_SUPER'], 1,  params['NUM_SAMPLES_SUPER'])
workers3 = test3(model, params['NUM_CASES_SUPER'], 1,  params['NUM_SAMPLES_SUPER'])
logging.info("[{0}, {1}, {2}]".format(workers1/params['NUM_CASES_SUPER'],
    workers2/params['NUM_CASES_SUPER'], workers3/params['NUM_CASES_SUPER']))

t0 = time.time()
tprev = time.time()
#Main training loop over super epoch
try:
    for x in range(params['SUPER_EPOCHS']):
        logging.info("Started at: " + time.strftime("%a, %d %b %Y %I:%M:%S %p", time.localtime()))
        logging.info('Learning Rate:{0}, Weight_Decay:{1}, k={2}'
            .format(model.learning_rate, model.weight_decay, model.k))
        err = model.train(train_loader, params['SUB_EPOCHS'], test_fn=test, pcd=params['PCD'])

        fa_errors.append(err[0])
        test_errors.append(err[1])

        tnow = time.time()
        logging.info('time for this iteration=' + str(tnow - tprev))
        tprev = time.time()


        logging.info("SUPER EPOCH FINISHED, CHANGING MODEL PARAMETERS, PERFORMING FULL TEST")
        model.k = int(model.k * params['CDK_GROWTH_RATE'])
        model.learning_rate = model.learning_rate/params['LEARNING_DECAY']
        model.weight_decay = model.weight_decay/params['LEARNING_DECAY']

        if params['type'] == 'Mult':
            logging.info("[Mult, Div, Fact] num_cases={0}, num_samples={1}"
                .format(params['NUM_CASES_SUPER'], params['NUM_SAMPLES_SUPER']))
        else:
            logging.info("[Add, Sub, Rev] num_cases={0}, num_samples={1}"
                .format(params['NUM_CASES_SUPER'], params['NUM_SAMPLES_SUPER']))

        #testing after each super epoch is generally more intensive than sub epoch
        workers1 = test1(model, params['NUM_CASES_SUPER'], 1,  params['NUM_SAMPLES_SUPER'])
        workers2 = test2(model, params['NUM_CASES_SUPER'], 1,  params['NUM_SAMPLES_SUPER'])
        workers3 = test3(model, params['NUM_CASES_SUPER'], 1,  params['NUM_SAMPLES_SUPER'])
        logging.info("[{0}, {1}, {2}]".format(workers1/params['NUM_CASES_SUPER'],
            workers2/params['NUM_CASES_SUPER'], workers3/params['NUM_CASES_SUPER']))

        ########## Save it! #################
        model.cpu()
        outname = 'trained/' + file_name + '.p'
        RBM.save(model, outname)
        logging.info("Saved to: " + outname)
        if params['CUDA']:
            model.cuda(device=params['DEVICE'])

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

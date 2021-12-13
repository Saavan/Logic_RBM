#include <iostream>
#include <cublas_v2.h>
#include <curand.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "rbm.h"
#include "rbm_kernels.h"
#include "bytecont.h"
#include "utils.h"

#define INFO  true
#define DEBUG false

#define BLOCK_SIZE 64 // it is assumed that a number of training examples is big (exv and exh are bigger than 1024
                        // otherwise BLOCK_SIZE should be set dynamically

#define STRIDE 1024 //Number of samples to store on GPU before pushing to CPU

RBM::RBM(int visible, int hidden, float rate) {
    numVisible   = visible;
    numHidden    = hidden;
    learningRate = rate;

    int weightsNumber = (numVisible) * (numHidden); //vis*hid weights
    checkCudaError(__LINE__, cudaMalloc(&dWeights, weightsNumber * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dVisBias, numVisible * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dHidBias, numHidden * sizeof(float)));
    checkCuRandError(__LINE__, curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_XORWOW));
    checkCuRandError(__LINE__, curandGenerateNormal(generator, dWeights, weightsNumber, 0.0, 1.0));
    checkCuRandError(__LINE__, curandGenerateNormal(generator, dVisBias, numVisible, 0.0, 1.0));
    checkCuRandError(__LINE__, curandGenerateNormal(generator, dHidBias, numHidden, 0.0, 1.0));

    if (INFO) {
        std::cout << "Initial weights:" << std::endl;
        printDeviceColumnMajorMatrix(dWeights, numVisible, numHidden);
    }

    checkCuBlasError(__LINE__, cublasCreate(&handle));
    std::cout << "RBM initialized" << std::endl;
}

RBM::~RBM() {
    checkCudaError(__LINE__, cudaFree(dWeights));
    cublasDestroy(handle);
    curandDestroyGenerator(generator);
    std::cout << "RBM destroyed" << std::endl;
}

float *RBM::hiddenActivationProbabilities(float *dVisibleUnitsStates, int examplesNumber) {
    float *dHiddenUnitsActivationEnergy;        // matrix of float values of dim exh
    float *dHiddenUnitsActivationProbabilities; // matrix of [0,1] values of dim exh

    int hiddenBufferSize = numHidden * examplesNumber;

    checkCudaError(__LINE__, cudaMalloc(&dHiddenUnitsActivationEnergy, hiddenBufferSize * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dHiddenUnitsActivationProbabilities, hiddenBufferSize * sizeof(float)));

    if (DEBUG) std::cout << "Calculating hidden units activation energies" << std::endl;

    const float alpha = 1;
    const float beta  = 0;
    checkCuBlasError(__LINE__, cublasSgemm(
                         handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_N,
                         examplesNumber,
                         numHidden + 1,
                         numVisible + 1,
                         &alpha,
                         dVisibleUnitsStates,
                         examplesNumber, // lda
                         dWeights,
                         numVisible + 1, // ldb
                         &beta,
                         dHiddenUnitsActivationEnergy,
                         examplesNumber)); // ldc

    if (DEBUG) printDeviceColumnMajorMatrix(dHiddenUnitsActivationEnergy, examplesNumber, numHidden + 1);

    int blockNumber = hiddenBufferSize / BLOCK_SIZE + 1;
    if (DEBUG) std::cout << "Calculating hidden probabilities " << BLOCK_SIZE << std::endl;
    sigmoid<<<blockNumber, BLOCK_SIZE>>>(dHiddenUnitsActivationEnergy, dHiddenUnitsActivationProbabilities, hiddenBufferSize);
    checkCudaError(__LINE__);
    if (DEBUG) printDeviceColumnMajorMatrix(dHiddenUnitsActivationProbabilities, examplesNumber, numHidden + 1);

    checkCudaError(__LINE__, cudaFree(dHiddenUnitsActivationEnergy));
    return dHiddenUnitsActivationProbabilities;
}



float *RBM::visibleActivationProbabilities(float *dHiddenUnitsStates, int examplesNumber) {
    float *dVisibleUnitsActivationEnergy;        // matrix of float values of dim exv
    float *dVisibleUnitsActivationProbabilities; // matrix of [0,1] values of dim exv

    int visibleBufferSize = (numVisible + 1) * examplesNumber;

    checkCudaError(__LINE__, cudaMalloc(&dVisibleUnitsActivationEnergy, visibleBufferSize * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dVisibleUnitsActivationProbabilities, visibleBufferSize * sizeof(float)));

    if (DEBUG) std::cout << "Calculating visible units activation energies" << std::endl;

    const float alpha = 1;
    const float beta  = 0;
    checkCuBlasError(__LINE__, cublasSgemm(
                         handle,
                         CUBLAS_OP_N,
                         CUBLAS_OP_T,
                         examplesNumber,
                         numVisible + 1,
                         numHidden + 1,
                         &alpha,
                         dHiddenUnitsStates,
                         examplesNumber, // lda
                         dWeights,
                         numVisible + 1, // ldb
                         &beta,
                         dVisibleUnitsActivationEnergy,
                         examplesNumber)); // ldc

    if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationEnergy, examplesNumber, numVisible + 1);

    int blockNumber = visibleBufferSize / BLOCK_SIZE + 1;
    if (DEBUG) std::cout << "Calculating visible probabilities" << std::endl;

    if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);

    sigmoid<<<blockNumber, BLOCK_SIZE>>>(dVisibleUnitsActivationEnergy, dVisibleUnitsActivationProbabilities, visibleBufferSize);

    checkCudaError(__LINE__);

    if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);

    checkCudaError(__LINE__, cudaFree(dVisibleUnitsActivationEnergy));
    return dVisibleUnitsActivationProbabilities;
}

float *RBM::computeAssociations(float *dVisibleUnitsActivationProbabilities, float *dHiddenUnitsActivationProbabilities, int examplesNumber) {
    float *dAssociations; // vxh matrix

    checkCudaError(__LINE__, cudaMalloc(&dAssociations, (numVisible + 1) * (numHidden + 1) * sizeof(float))); // +1 because of bias

    const float alpha = 1;
    const float beta  = 0;
    checkCuBlasError(__LINE__, cublasSgemm(
                         handle,
                         CUBLAS_OP_T,
                         CUBLAS_OP_N,
                         numVisible + 1,
                         numHidden + 1,
                         examplesNumber,
                         &alpha,
                         dVisibleUnitsActivationProbabilities,
                         examplesNumber, // lda
                         dHiddenUnitsActivationProbabilities,
                         examplesNumber, // ldb
                         &beta,
                         dAssociations,
                         numVisible + 1)); // ldc

    if (DEBUG) std::cout << "Associations:" << std::endl;
    if (DEBUG) printDeviceColumnMajorMatrix(dAssociations, numVisible + 1, numHidden + 1);

    return dAssociations;
}

// a contrastive divergence (CD_1) learning algorithm; batched version
void RBM::train(float *hTrainingData, int examplesNumber, int maxEpochs) {
    float hBias[examplesNumber];                        // will be added as a first column of training data
    std::fill_n(hBias, examplesNumber, 1.0);

    float *dVisibleUnitsStates;                         // device copy of training data
    float *dVisibleUnitsActivationProbabilities;        // matrix of [0,1] of dimensions exv

    float *dHiddenUnitsStates;                          // matrix of boolean values of dimensions exh
    float *dPositiveHiddenUnitsActivationProbabilities; // matrix of [0,1] of dimensions exh

    float *dNegativeHiddenUnitsActivationProbabilities; // matrix of [0,1] of dimensions exh

    float *dPositiveAssociations;                       // matrix of dimensions vxh
    float *dNegativeAssociations;                       // matrix of dimensions vxh

    float *dRandom;                                     // matrix of dimensions exh of random values [0,1]

    int visibleBufferSize = (numVisible) * examplesNumber;
    int hiddenBufferSize  = (numHidden) * examplesNumber;

    checkCudaError(__LINE__, cudaMalloc(&dVisibleUnitsStates, visibleBufferSize * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dHiddenUnitsStates, hiddenBufferSize * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dRandom, hiddenBufferSize * sizeof(float)));

    for (int e = 0; e < maxEpochs; e++) {
        // a positive phase of the contrastive divergence

        // copy bias to the first column
        checkCudaError(__LINE__, cudaMemcpy(dVisibleUnitsStates, hBias, examplesNumber * sizeof(float), cudaMemcpyHostToDevice));

        // copy training data to remaining cells
        checkCudaError(__LINE__, cudaMemcpy(&dVisibleUnitsStates[examplesNumber],
                                            hTrainingData,
                                            numVisible * examplesNumber * sizeof(float),
                                            cudaMemcpyHostToDevice));

        if (DEBUG) std::cout << "Visible units states:" << std::endl;
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsStates, examplesNumber, numVisible + 1);

        // calculate positive hidden activation probabilities

        dPositiveHiddenUnitsActivationProbabilities = hiddenActivationProbabilities(dVisibleUnitsStates, examplesNumber);

        if (DEBUG) std::cout << "Fixing hidden units activation probabilities by setting bias to the first column" << std::endl;
        checkCudaError(__LINE__, cudaMemcpy(dPositiveHiddenUnitsActivationProbabilities, hBias, examplesNumber * sizeof(float), cudaMemcpyHostToDevice));
        if (DEBUG) printDeviceColumnMajorMatrix(dPositiveHiddenUnitsActivationProbabilities, examplesNumber, numHidden + 1);

        if (DEBUG) std::cout << "Calculating hidden unit states by sampling" << std::endl;
        checkCuRandError(__LINE__, curandGenerateUniform(generator, dRandom, hiddenBufferSize));
        int blockNumber = hiddenBufferSize / BLOCK_SIZE + 1;
        greaterThan<<<blockNumber, BLOCK_SIZE>>>(dPositiveHiddenUnitsActivationProbabilities, dRandom, dHiddenUnitsStates, examplesNumber * (numHidden + 1));
        checkCudaError(__LINE__);
        if (DEBUG) printDeviceColumnMajorMatrix(dHiddenUnitsStates, examplesNumber, numHidden + 1);

        dPositiveAssociations = computeAssociations(dVisibleUnitsStates, dPositiveHiddenUnitsActivationProbabilities, examplesNumber);

        // a negative (reconstruction) phase of the contrastive divergence

        // calculate negative visible probabilities
        dVisibleUnitsActivationProbabilities = visibleActivationProbabilities(dHiddenUnitsStates, examplesNumber);

        if (DEBUG) std::cout << "Visible Units Activation Probabilities:" << std::endl;
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);

        if (DEBUG) std::cout << "Fixing visible units activation probabilities by setting bias to the first column" << std::endl;
        checkCudaError(__LINE__, cudaMemcpy(dVisibleUnitsActivationProbabilities, hBias, examplesNumber * sizeof(float), cudaMemcpyHostToDevice));
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsActivationProbabilities, examplesNumber, numVisible + 1);

        // negative hidden probabilities
        dNegativeHiddenUnitsActivationProbabilities = hiddenActivationProbabilities(dVisibleUnitsActivationProbabilities, examplesNumber);

        if (DEBUG) std::cout << "Negative Hidden units activation probabilities:" << std::endl;
        if (DEBUG) printDeviceColumnMajorMatrix(dNegativeHiddenUnitsActivationProbabilities, examplesNumber, numHidden + 1);

        if (DEBUG) std::cout << "Calculating negative associations" << std::endl;
        dNegativeAssociations = computeAssociations(dVisibleUnitsActivationProbabilities, dNegativeHiddenUnitsActivationProbabilities, examplesNumber);

        if (DEBUG) std::cout << "Updating weights" << std::endl;
        int weightsNumber = (numHidden + 1) * (numVisible + 1);
        blockNumber = weightsNumber / BLOCK_SIZE + 1;
        //updateWeight<<<blockNumber, BLOCK_SIZE>>>(dWeights, dPositiveAssociations, dNegativeAssociations, weightsNumber, examplesNumber, learningRate);
        checkCudaError(__LINE__);
        if (DEBUG) printDeviceColumnMajorMatrix(dWeights, numVisible + 1, numHidden + 1);

        blockNumber = visibleBufferSize / BLOCK_SIZE + 1;
        if (DEBUG) std::cout << "Calculating error - squares of subtractions: " << std::endl;

        // for memory efficiency we will write subtraction result to one of the input matrices (dVisibleUnitsStates)
        subAndSquare<<<blockNumber, BLOCK_SIZE>>>(dVisibleUnitsStates, dVisibleUnitsActivationProbabilities, visibleBufferSize);
        checkCudaError(__LINE__);
        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsStates, examplesNumber, numVisible + 1);

        if (DEBUG) std::cout << "Calculation error - reducing sum:" << std::endl;
        thrust::device_ptr<float> dVisibleUnitsStatesPtr(dVisibleUnitsStates);
        float hError = thrust::reduce(dVisibleUnitsStatesPtr, dVisibleUnitsStatesPtr + visibleBufferSize, 0.0, thrust::plus<float>());

        if (DEBUG) printDeviceColumnMajorMatrix(dVisibleUnitsStates, examplesNumber, numVisible + 1);

        std::cout << "Reconstruction error after epoch " << e + 1 << " is " << hError << std::endl;

        checkCudaError(__LINE__, cudaFree(dVisibleUnitsActivationProbabilities));
        checkCudaError(__LINE__, cudaFree(dPositiveHiddenUnitsActivationProbabilities));
        checkCudaError(__LINE__, cudaFree(dNegativeHiddenUnitsActivationProbabilities));
        checkCudaError(__LINE__, cudaFree(dPositiveAssociations));
        checkCudaError(__LINE__, cudaFree(dNegativeAssociations));
    }

    checkCudaError(__LINE__, cudaFree(dRandom));
    checkCudaError(__LINE__, cudaFree(dVisibleUnitsStates));
    checkCudaError(__LINE__, cudaFree(dHiddenUnitsStates));

    if (INFO) std::cout << "Learned weights:" << std::endl;
    if (INFO) printDeviceColumnMajorMatrix(dWeights, numVisible + 1, numHidden + 1);
}

float *RBM::hiddenStates(float *hVisible) {
    float *dVisible;
    float *dHidden;
    float *hHidden;
    float *dRandom;

    checkCudaError(__LINE__, cudaMalloc(&dVisible, (numVisible + 1) * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dHidden, (numHidden + 1) * sizeof(float)));

    float bias = 1.0;
    checkCudaError(__LINE__, cudaMemcpy(dVisible, &bias, sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(__LINE__, cudaMemcpy(&dVisible[1], hVisible, numVisible * sizeof(float), cudaMemcpyHostToDevice)); // set bias

    dHidden = hiddenActivationProbabilities(dVisible, 1);

    // sampling
    checkCudaError(__LINE__, cudaMalloc(&dRandom, (numHidden + 1) * sizeof(float)));
    checkCuRandError(__LINE__, curandGenerateUniform(generator, dRandom, numHidden + 1));
    int blockNumber = (numHidden + 1) / BLOCK_SIZE + 1;
    greaterThan<<<blockNumber, BLOCK_SIZE>>>(dHidden, dRandom, dHidden, numHidden + 1);
    checkCudaError(__LINE__);

    hHidden = (float *) malloc(numHidden * sizeof(float));
    cudaMemcpy(hHidden, &dHidden[1], numHidden * sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaError(__LINE__, cudaFree(dHidden));
    checkCudaError(__LINE__, cudaFree(dVisible));
    checkCudaError(__LINE__, cudaFree(dRandom));
    return hHidden;
}
void RBM::sampHidden(const float *dVisStates, float *dHidActivation) {

    if (DEBUG) std::cout << "Calculating hidden units activation energies" << std::endl;

    const float alpha = 1;
    const float beta  = 0;
    // Vector matrix multiply 
    checkCuBlasError(__LINE__, cublasSgemv(
                         handle,
                         CUBLAS_OP_N,
                         numVisible,
                         numHidden,
                         &alpha,
                         dWeights, numHidden, // lda
                         dVisStates, 1, // incx
                         &beta,
                         dHidActivation, 1)); // incy

    // Summation
    checkCuBlasError(__LINE__, cublasSaxpy(
                handle,
                numHidden,
                &alpha,
                dHidBias, 1,
                dHidActivation, 1));


    if (DEBUG) printDeviceColumnMajorMatrix(dHidActivation, 1, numHidden);

    int blockNumber = numHidden / BLOCK_SIZE + 1;
    if (DEBUG) std::cout << "Calculating hidden probabilities " << BLOCK_SIZE << std::endl;
    sigmoid<<<blockNumber, BLOCK_SIZE>>>(dHidActivation, dHidActivation, numHidden);
    if (DEBUG) printDeviceColumnMajorMatrix(dHidActivation, 1 , numHidden);
}


void RBM::sampVisible(const float *dHidStates, float *dVisActivation) {

    if (DEBUG) std::cout << "Calculating hidden units activation energies" << std::endl;

    const float alpha = 1;
    const float beta  = 0;
    // Vector matrix multiply 
    checkCuBlasError(__LINE__, cublasSgemv(
                         handle,
                         CUBLAS_OP_T,
                         numVisible,
                         numHidden,
                         &alpha,
                         dWeights, numHidden, // lda
                         dHidStates, 1, // incx
                         &beta,
                         dVisActivation, 1)); // incy

    // Summation
    checkCuBlasError(__LINE__, cublasSaxpy(
                handle,
                numVisible,
                &alpha,
                dVisBias, 1,
                dVisActivation, 1));


    if (DEBUG) printDeviceColumnMajorMatrix(dVisActivation, 1, numVisible);

    int blockNumber = numVisible / BLOCK_SIZE + 1;
    if (DEBUG) std::cout << "Calculating hidden probabilities " << BLOCK_SIZE << std::endl;
    sigmoid<<<blockNumber, BLOCK_SIZE>>>(dVisActivation, dVisActivation, numVisible);
    if (DEBUG) printDeviceColumnMajorMatrix(dVisActivation, 1 , numVisible);
}

bytemap *RBM::genStats(int numSamps) {
    float *dVisible;
    float *dHidden;
    float *dvisRandom;
    float *dhidRandom;
    float *dSamps;
    float *dSamps1;
    float *dSamps2;
    float *hSamps;
    
    bytemap *hashmap = new bytemap(); 

    int blockNumberVis = (numVisible) / BLOCK_SIZE + 1;
    int blockNumberHid = (numHidden) / BLOCK_SIZE + 1;
    
    checkCudaError(__LINE__, cudaMalloc(&dVisible, (numVisible) * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dHidden, (numHidden) * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dSamps1,  numVisible * STRIDE * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dSamps2,  numVisible * STRIDE * sizeof(float)));

    hSamps = new float[numVisible*STRIDE];
     
    checkCudaError(__LINE__, cudaMalloc(&dvisRandom, (numVisible) * sizeof(float)));
    checkCudaError(__LINE__, cudaMalloc(&dhidRandom, (numVisible) * sizeof(float)));
    checkCuRandError(__LINE__, curandGenerateUniform(generator, dvisRandom, numVisible));
    checkCuRandError(__LINE__, curandGenerateUniform(generator, dhidRandom, numVisible));
    greaterThan<<<blockNumberVis, BLOCK_SIZE>>>(dvisRandom, dhidRandom, dVisible, numVisible);
    checkCudaError(__LINE__);
    checkCudaError(__LINE__, cudaFree(dhidRandom));
    checkCudaError(__LINE__, cudaMalloc(&dhidRandom, (numHidden) * sizeof(float)));
    
    // sampling
    ByteCont key(numVisible);
    for(int i = 0; i < numSamps; i += STRIDE) { 
        if(i%2 == 0) {
            dSamps = dSamps1;
        } else {
            dSamps = dSamps2;
        }
        for(int j = 0; j < STRIDE; j++) {
            checkCuRandError(__LINE__, curandGenerateUniform(generator, dvisRandom, numVisible));
            checkCuRandError(__LINE__, curandGenerateUniform(generator, dhidRandom, numHidden));
            sampHidden(dVisible, dHidden);
            greaterThan<<<blockNumberHid, BLOCK_SIZE>>>(dHidden, dhidRandom, dHidden, numHidden);
            checkCudaError(__LINE__);
            sampVisible(dHidden, dVisible);
            greaterThan<<<blockNumberVis, BLOCK_SIZE>>>(dVisible, dvisRandom, dVisible, numVisible);
            checkCudaError(__LINE__);
            checkCudaError(__LINE__, cudaMemcpyAsync(dSamps + j*numVisible, dVisible, 
                        numVisible*sizeof(float), cudaMemcpyDeviceToDevice));
            if (DEBUG) std::cout << "Current Index " << j*numVisible << std::endl;
        }
        
        checkCudaError(__LINE__, cudaMemcpy(hSamps, dSamps, numVisible*STRIDE*sizeof(float), cudaMemcpyDeviceToHost));
        
        for(unsigned long k = 0; k < STRIDE; k++) {
            key.setCont(hSamps + k*numVisible);
            auto search = hashmap->find(key);
            if (search == hashmap->end()) {
                hashmap->insert({key, 1});
            }
            else {
                search->second += 1;
            }
        }
        
    }



    checkCudaError(__LINE__, cudaFree(dHidden));
    checkCudaError(__LINE__, cudaFree(dVisible));
    checkCudaError(__LINE__, cudaFree(dSamps1));
    checkCudaError(__LINE__, cudaFree(dSamps2));
    checkCudaError(__LINE__, cudaFree(dvisRandom));
    checkCudaError(__LINE__, cudaFree(dhidRandom));
    return hashmap;
}

#ifndef RBM_H_
#define RBM_H_

#include <cublas_v2.h>
#include <curand.h>
#include "bytecont.h"

class RBM {
 private:
    int    numVisible;
    int    numHidden;
    float  learningRate;
    float *dWeights; // column-major matrix of dim numVisible x numHidden
    float *dVisBias; // vector of dim numVisible
    float *dHidBias; // vector of dim numHidden
    cublasHandle_t    handle;
    curandGenerator_t generator;

    float *hiddenActivationProbabilities(float *dVisibleUnitsStates, int examplesNumber);
    float *visibleActivationProbabilities(float *dHiddenUnitsStates, int examplesNumber);
    float *computeAssociations(float *dVisibleUnitsActivationProbabilities,
                               float *dHiddenUnitsActivationProbabilities,
                               int    examplesNumber);

 public:
    RBM(int visibleNumber, int hiddenNumber, float rate);
    ~RBM();
    // hTrainingData is a matrix of BOOLEAN values (true is represented as 1; false is represented by 0)
    // each row is a training example consisting of the states of visible units
    // matrix is written to array in column-major order
    void train(float *hTrainingData, int examplesNumber, int maxEpochs);
    float *hiddenStates(float *hVisible);
    void sampHidden(const float *dVisStates, float *dHidActivation);
    void sampVisible(const float *dHidStates, float *dVisActivation);
    bytemap *genStats(int numSamps);
    

};

#endif  // RBM_H_

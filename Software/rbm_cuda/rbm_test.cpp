#include <iostream>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include "rbm.h"
#include "utils.h"
#include "bytecont.h"
#include <chrono> 

using namespace std;

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

float *loadMNISTDataSet(string fileName) {
    string filePath("mnist/");

    filePath.append(fileName);
    ifstream file(filePath.c_str());

    float *mnistDataSet;
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *) &magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char *) &number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char *) &n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char *) &n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        cout << "Loading " << number_of_images << " images of dimensions " << n_rows << "x" << n_cols << " from " << filePath << endl;

        mnistDataSet = (float *) malloc(n_rows * n_cols * number_of_images * sizeof(float));

        unsigned char temp = 0;
        float flag = 0;
        int   pixelIdx;
        int   index;
        for (int e = 0; e < number_of_images; e++) {
            for (int r = 0; r < n_rows; r++) {
                for (int c = 0; c < n_cols; c++) {
                    file.read((char *) &temp, sizeof(temp));
                    flag = temp > 127;  // binarization
                    pixelIdx = r * n_cols + c;
                    index    = pixelIdx * number_of_images + e; // MNIST images are written row-major; CuBLAS matrices
                                                                // need to be column-major
                    if (flag)
                        mnistDataSet[index] = 1;
                    else
                        mnistDataSet[index] = 0;
                }
            }
        }
        cout << "Images loaded into RAM" << endl;
    }

    return mnistDataSet;
}

void printMnistImages(int numberOfImagesToPring) {
    float *mnistDataSet = loadMNISTDataSet("train-images-idx3-ubyte");

    int pixelIdx;
    int index;

    for (int e = 0; e < numberOfImagesToPring; e++) {
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                pixelIdx = r * 28 + c;
                index    = pixelIdx * 60000 + e;
                if (mnistDataSet[index] == 1)
                    cout << setw(2) << "X";
                else
                    cout << setw(2) << "";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void testMNIST() {
    RBM rbm(28 * 28, 500, 0.1);

    float *trainingData = loadMNISTDataSet("train-images-idx3-ubyte");

    rbm.train(trainingData, 60000, 100);
}

void testSmallDataset() {
    RBM rbm(6, 2, 0.1);
    // column-major version of matrix:
    // [1,1,1,0,0,0,
    //  1,0,1,0,0,0,
    //  1,1,1,0,0,0,
    //  0,0,1,1,1,0,
    //  0,0,1,1,0,0]
    float trainingData[] = {
        1, 1, 1, 0, 0, 0,
        1, 0, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1,
        0, 0, 0, 1, 0, 1,
        0, 0, 0, 0, 0, 0
    };

    cout << "Training data:" << endl;
    printColumnMajorMatrix(trainingData, 6, 6);
    rbm.train(trainingData, 6, 100);

    float visible1 [] = {0, 0, 0, 1, 1, 0};
    cout << "Visible=" << endl;
    printColumnMajorMatrix(visible1, 1, 6);
    float *hidden = rbm.hiddenStates(visible1);
    cout << "Hidden=" << endl;
    printColumnMajorMatrix(hidden, 1, 2);

    float visible2 [] = {1, 0, 0, 0, 0, 0};
    cout << "Visible=" << endl;
    printColumnMajorMatrix(visible2, 1, 6);
    hidden = rbm.hiddenStates(visible2);
    cout << "Hidden=" << endl;
    printColumnMajorMatrix(hidden, 1, 2);

    float visible3 [] = {0, 0, 0, 0, 0, 1};
    cout << "Visible=" << endl;
    printColumnMajorMatrix(visible3, 1, 6);
    hidden = rbm.hiddenStates(visible3);
    cout << "Hidden=" << endl;
    printColumnMajorMatrix(hidden, 1, 2);
}

void testSamp() {
    bytemap* outSamps;
    RBM rbm(6, 20, 0.1);
    outSamps = rbm.genStats(100);
    //printColumnMajorMatrix(outSamps, 100, 6);
}

void testBench() {
    bytemap* outSamps;
    int numSamps = 1000000;
    RBM Mult4 = RBM(8, 16, 0.1);
    RBM Mult6 = RBM(12, 48, 0.1);
    RBM Mult8 = RBM(16, 64, 0.1);
    RBM Mult10 = RBM(20, 144, 0.1);
    RBM Mult12 = RBM(50, 352, 0.1);
    RBM Mult16 = RBM(78, 576, 0.1);
    auto start = std::chrono::high_resolution_clock::now();
    outSamps = Mult4.genStats(numSamps);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end-start;
    std::cout << "Mult4 elapsed time: " << elapsed.count()/numSamps << "s per Sample" << endl;

    start = std::chrono::high_resolution_clock::now();
    outSamps = Mult6.genStats(numSamps);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end-start;
    std::cout << "Mult6 elapsed time: " << elapsed.count()/numSamps << "s per Sample" << endl;
    
    start = std::chrono::high_resolution_clock::now();
    outSamps = Mult8.genStats(numSamps);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end-start;
    std::cout << "Mult8 elapsed time: " << elapsed.count()/numSamps << "s per Sample" << endl;
        
    start = std::chrono::high_resolution_clock::now();
    outSamps = Mult10.genStats(numSamps);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end-start;
    std::cout << "Mult10 elapsed time: " << elapsed.count()/numSamps << "s per Sample" << endl;
    
    start = std::chrono::high_resolution_clock::now();
    outSamps = Mult12.genStats(numSamps);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end-start;
    std::cout << "Mult12 elapsed time: " << elapsed.count()/numSamps << "s per Sample" << endl;
    
    start = std::chrono::high_resolution_clock::now();
    outSamps = Mult16.genStats(numSamps);
    end = std::chrono::high_resolution_clock::now();
    elapsed = end-start;
    std::cout << "Mult16 elapsed time: " << elapsed.count()/numSamps << "s per Sample" << endl;
}

int main(int argc, char **argv) {
    if (argc != 2)
        return 1;

    string arg(argv[1]);

    if (arg.compare(string("mnist")) == 0) {
        testMNIST();
        return 0;
    } else if (arg.compare(string("small")) == 0) {
        testSmallDataset();
        return 0;
    } else if (arg.compare(string("samp")) == 0) { 
        testBench();
        return 0;
    } else if (arg.compare(string("print")) == 0) {
        printMnistImages(5);
        return 0;
    }
    return 1;
}

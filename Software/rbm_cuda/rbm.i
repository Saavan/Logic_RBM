%module rbm
%{
    #include "rbm.h"
%}

%include "std_string.i"
%include "std_map.i"
%include "std_unordered_map.i"
%include "std_vector.i"
%include "carrays.i"

/* Creates functions for working with int arrays */
%array_class(int, intArr);

/* typedef std::unordered_map<ByteCont, int, ByteHasher> bytemap; */

/*
%template(samp_map) std::unordered_map<ByteCont, int, ByteHasher>;
%template(iVec) std::vector<int>;
%template(bVec) std::vector<bool>;
%template(byteVec) std::vector<ByteCont>;

 This allows for Python hashing of the ByteCont type 
%rename(__hash__) ByteCont::getHash;
%feature("python:slot", "tp_hash", functype="hashfunc") ByteCont::getHash;
*/
/*
struct ByteCont {
    char * buf;
    int buf_len;
    long getHash();
    ByteCont();
    ByteCont(int buf_len);
    ByteCont(const ByteCont& other);
    ~ByteCont();
    bool operator<(const ByteCont& rhs) const; 
    bool operator==(const ByteCont& rhs) const; 
};
*/
class RBM {
    RBM(int visibleNumber, int hiddenNumber, float rate);
    ~RBM();
    void train(float *hTrainingData, int examplesNumber, int maxEpochs);
    float *hiddenStates(float *hVisible);
    void sampHidden(const float *dVisStates, float *dHidActivation);
    void sampVisible(const float *dHidStates, float *dVisActivation);
    bytemap *genStats(int numSamps);
};
/*
std::string convert(const ByteCont* b);
std::string convert(const ByteCont* b, int num_visible, int* outbits, size_t outbits_len);
std::vector<int> convert_int(const ByteCont* b);
std::vector<int> convert_int(const ByteCont* b, int num_visible, int* outbits, size_t outbits_len);


void read_samps(bytemap* hashmap, unsigned long num_samps, int buf_len);
void read_samps(bytemap* hashmap, unsigned long num_samps, int buf_len, int num_visible, int* outbits, size_t outbits_len);
std::vector<ByteCont>* read_samps_raw(unsigned long num_samps, int buf_len);
ByteCont* read_samps_hit(unsigned long num_samps, int buf_len);
ByteCont* read_samps_hit(unsigned long num_samps, int buf_len, int num_visible, int* outbits, size_t outbits_len);
const ByteCont* argmax(const bytemap* hashmap);
void del_map(bytemap* hashmap);

std::vector<int> inttovec(int* x);
void print_int(const int* x, size_t len);
void print_intvec(std::vector<int> invec);

std::string decimal_to_fixed(double val, int n, int p);
*/

#include <stdio.h>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>
#include <iostream>

#ifndef BYTECONT_H_
#define BYTECONT_H_

int intcomp(int lhs[4], int rhs[4]);


struct int_classcomp {
  bool operator() (int lhs[4], int rhs[4]) const
  {return intcomp(lhs, rhs) < 0;}
};

struct str_classcomp {
    bool operator() (char lhs[16], char rhs[16]) const
    {return strncmp(lhs, rhs, 16) < 0;}
};


struct ByteCont {
    char * buf;
    int buf_len;
    long getHash();
    ~ByteCont();
    ByteCont();
    ByteCont(int buf_len);
    ByteCont(float *samp, int buf_len);
    ByteCont(const ByteCont& other);
    bool operator<(const ByteCont& rhs) const; 
    bool operator==(const ByteCont& rhs) const; 
    void setCont(const float *samp);
};

struct ByteHasher {
    size_t operator()(const ByteCont& b) const;
};


typedef std::unordered_map<ByteCont, int, ByteHasher> bytemap;

const ByteCont* argmax(const bytemap* hashmap);

#endif  // BYTECONT_H_

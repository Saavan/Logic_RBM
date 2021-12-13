#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <fcntl.h>
#include <map>
#include <iostream>
#include <string>
#include <string.h>
#include <bitset>
#include <vector>
#include "bytecont.h"
using namespace std;

//Just the default constructor (needed for swig for some reason?)
ByteCont::ByteCont() {}
/*
 * This constructor should be used whenever making a ByteCont
 */
ByteCont::ByteCont(int buf_len) {
    this->buf_len = buf_len;
    this->buf = new char[buf_len];
}

ByteCont::ByteCont(float *samp, int buf_len) {
    this->buf_len = buf_len;
    this->buf = new char[buf_len];
    for(int i = 0; i < buf_len; i++) {
        buf[i] = samp[i]; 
    }
}


/*
 * Copy Constructor
 */
ByteCont::ByteCont(const ByteCont& other) :
buf_len(other.buf_len) 
{
   this->buf = new char[buf_len]; 
   memcpy(this->buf, other.buf, this->buf_len);
}



/*
 * Byte Containers are just raw memory containers, standard C functions apply
 */
bool ByteCont::operator<(const ByteCont& rhs) const {
    return memcmp(this->buf, rhs.buf, this->buf_len) <= 0; 
}
bool ByteCont::operator==(const ByteCont& rhs) const {
    return memcmp(this->buf, rhs.buf, this->buf_len) == 0;
}

ByteCont::~ByteCont() {
    delete[] this->buf; 
}

/*
 * Hash function for use with STL unordered_map
 * Effectively a copy of the boost map
 */
size_t ByteHasher::operator()(const ByteCont& b) const {
    size_t state = b.buf_len;
    const size_t seed = 0x9e3779b9; //Pick random internal state
    int len = sizeof(size_t);
    int offset = 0;
    for(int i = 0; i < b.buf_len/len; i++) {
        //cheesy hack to convert contiguous values into the correct type
        size_t val = *(size_t *) &b.buf[offset];
        //Additions are better than xor for combining state
        state ^= val + seed + (state << 6) + (state >> 3); 
        offset += len;
    }
    return state;
}
/*
 * Used with swig/python so asdict() function works properly
 */
long ByteCont::getHash() {
    ByteHasher test;
    return (long) test(*this);
}

void ByteCont::setCont(const float *samp){
    for(int i = 0; i < buf_len; i++){
        buf[i] = samp[i];
    }
}

/*
 * Converts to a bitstring of 0s and 1s 
 * Fixes xillybus ordering weirdness, expects sequential 32 bit transations
 */
string convert(const ByteCont* b) {
    int bin_len = 8*b->buf_len;
    int buf_ind;
    int ind = 0;
    string out(bin_len, 0x30);
    
    for(int i = 0; i < b->buf_len/4; i++) {
        for(int j = 3; j >= 0; j--) {
            buf_ind = i*4 + j;
            for(int k = 7; k >= 0; k--) {
                //0x30 is UTF/ASCII 0 and 0x31 is UTF/ASCII 1
                out[ind] = 0x30 + abs((b->buf[buf_ind] >> k) % 2);
                ind++;
            }
        }
    }

    return out;
}
/*
 * Converts to a bitstring of 0s and 1s 
 * Takes a number of visible nodes, and outbits and only masks out those specific bit positions
 */
string convert(const ByteCont* b, int num_visible, int* outbits, size_t outbits_len) {
    string out(outbits_len, 0x30);

    int bit_tot, byte_loc, bit_loc;
    for(size_t i = 0; i < outbits_len; i++) {
        bit_tot = (num_visible - 1) - outbits[i];
        byte_loc = b->buf_len - (bit_tot / 8 + 1);
        //fixes stupid ordering issue with xillybus
        byte_loc = 4*(byte_loc/4) + (3 - byte_loc%4);
        bit_loc = bit_tot%8;
        //0x30 is UTF/ASCII 0 and 0x31 is UTF/ASCII 1
        out[i] = 0x30 + abs((b->buf[byte_loc] >> bit_loc) % 2); 
    }
    return out;
}
/*
 * Converts to a vector of 0s and 1s of integer type
 * Basically the same as above, but with a different output format
 */ 
vector<int> convert_int(const ByteCont* b) {
    vector<int> out(b->buf_len * 8);
    int buf_ind;
    auto data =  out.data();
    int ind = 0;
    //This is rather strange:
    //The Xillybus interface reads 32 bit chunks and bitwise flips them
    //We operate in the forward direction for 32 bit chunks
    // and flip elements within the chunk
    for(int i = 0; i < b->buf_len/4; i++) {
        for(int j = 3; j >= 0; j--) {
            buf_ind = i*4 + j;
            for(int k = 7; k >= 0; k--) {                
                data[ind] = abs((b->buf[buf_ind] >> k) % 2);
                ind++;
            }
        }
    }
    return out;
}
/*
 * Converts to a vector of 0s and 1s of integer type
 * takes in outbits of bit positions to mask out
 */ 
vector<int> convert_int(const ByteCont* b, int num_visible, int* outbits, size_t outbits_len) {
    vector<int> out(outbits_len);
    auto data =  out.data();

    int bit_tot, byte_loc, bit_loc;
    for(size_t i = 0; i < outbits_len; i++) {
        bit_tot = num_visible - 1 - outbits[i];
        byte_loc = b->buf_len - (bit_tot / 8 + 1);
        //fixes stupid ordering issue with xillybus
        byte_loc = 4*(byte_loc/4) + (3 - byte_loc%4);
        //bit_loc = 7 - bit_tot%8;
        bit_loc = bit_tot%8;
        data[i] = abs((b->buf[byte_loc] >> bit_loc) % 2); 
    }
    return out;
}


/*
 * Same as above, except creates a vector of booleans instead
 */
vector<bool> convert_bool(const ByteCont* b) {
    vector<bool> out(b->buf_len * 8);
    int buf_ind;
    auto it =  out.begin();
    //This is rather strange:
    //The Xillybus interface reads 32 bit chunks and flips them
    //from what you would expect
    for(int i = 0; i < b->buf_len/4; i++) {
        for(int j = 3; j >= 0; j--) {
            buf_ind = i*4 + j;
            for(int k = 7; k >= 0; k--) {                
                *it = (b->buf[buf_ind] >> (k)) % 2;
                it++;
            }
        }
    }
    return out;
}
/*
 * Shamelessly copied from geeksforgeeks.org
 */ 
string twos_comp(string str) 
{ 
    int n = str.length(); 
  
    // Traverse the string to get first '1' from 
    // the last of string 
    int i;
    for (i = n-1 ; i >= 0 ; i--) 
        if (str[i] == '1') 
            break; 
  
    // If there exists no '1' concatenate 1 at the 
    // starting of string 
    if (i == -1) 
        return '1' + str; 
  
    // Continue traversal after the position of 
    // first '1' 
    for (int k = i-1 ; k >= 0; k--) 
    { 
        //Just flip the values 
        if (str[k] == '1') 
            str[k] = '0'; 
        else
            str[k] = '1'; 
    } 
  
    // return the modified string 
    return str;
} 

/*
 * Converts a decimal number (in floating point format) into a fixed point bit array
 * with n bits and a point at position p
 * Returns a bitstring of 0s and 1s 
 */
string decimal_to_fixed(double val, int n, int p) {
    string out;
    bool is_neg;
    if (val < 0) {
        is_neg = true;
        val = -1 * val;
    } else {
        is_neg = false;
    }
    double residual = val;
    double lsb = exp2(-1 * p);
    double bit_val;
    for(int i = 0; i < n-1; i++){
        bit_val = exp2(n-2-i-p);
        if(bit_val <= residual + lsb/2) {
            out += '1';
            residual -= bit_val;
        }
        else 
            out += '0';
    }
    if(is_neg)
        return '1' + twos_comp(out);
    else
        return '0' + out;
}

/*
 * Fast way of finding the argmax of a hashmap 
 * Sorts by second argument of integer type
 */
const ByteCont* argmax(const bytemap* hashmap) {
    const ByteCont* max = &(hashmap->begin()->first);
    int maxval = hashmap->begin()->second;
    for(auto it = hashmap->begin(); it != hashmap->end(); ++it) {
        if(it->second > maxval) {
            max = &(it->first);
            maxval = it->second; 
        }   
    }
    return max;
}

void print_int(const int* x, size_t len) {
    cout << "[";
    for(unsigned int i = 0; i < len; i++){
        cout << x[i] << ", ";
    }
    cout << "]";

}

int intcomp(int lhs[4], int rhs[4]) {
    for(int i = 0; i < 4; i++) {
        if(!(lhs[i] == rhs[i])) return lhs[i] - rhs[i];
    }
    return 0;
}
/*
 * Main function that this file supports
 * reads samples from the xillybus address and puts them into a map
 * buf_len is how many bytes are in the xillybus FIFO
 */
void read_samps(bytemap* hashmap, unsigned long num_samps, int buf_len)
{
    FILE* read_fd = fopen("/dev/xillybus_read_32", "rb");
    if (read_fd==NULL) {fputs ("File error",stderr); exit (1);}
    int read_bytes = 0;
    
    ByteCont key(buf_len);
    for(unsigned long i = 0; i < num_samps; i++) {
        read_bytes += fread(key.buf, 1, buf_len, read_fd);
        auto search = hashmap->find(key);
        if (search == hashmap->end()) {
            hashmap->insert({key, 1});
        }
        else {
            search->second += 1;
        }
    }
    if (ferror(read_fd))
        cout << "Error:" << ferror(read_fd);
    fclose(read_fd);
}
/*
 * Similar function as above, but adds functionality for only looking at certain bit positions (specified by outbits)
 * num_visible is the number of bits in the model on the FPGA, specifies the bit offset to start looking for outbits
 * outbits is an integer array that tells positions to look at
 */
void read_samps(bytemap* hashmap, unsigned long num_samps, int buf_len, int num_visible, int* outbits, size_t outbits_len) {
    FILE* read_fd = fopen("/dev/xillybus_read_32", "rb");
    if (read_fd==NULL) {fputs ("File error",stderr); exit (1);}
    int read_bytes = 0;
    ByteCont key(buf_len);
   
    //Need extra () to ensure buf_mask is zeroed out
    char * buf_mask = new char[buf_len]();
    //Creates a buffer mask so that the hash only looks at the bits specified in the outbits array
    int bit_tot, byte_loc, bit_loc;
    for(size_t i = 0; i < outbits_len; i++) {
        bit_tot = num_visible - 1 - outbits[i];
        byte_loc = buf_len - (bit_tot / 8 + 1);
        //this should do conversion for xillybus byte indices
        byte_loc = 4*(byte_loc/4) + (3 - (byte_loc%4));
        bit_loc = bit_tot%8;
        buf_mask[byte_loc] = buf_mask[byte_loc] | (1 << bit_loc);  
    }
    
    for(unsigned long i = 0; i < num_samps; i++) {
        read_bytes += fread(key.buf, 1, buf_len, read_fd);
        //Masking appropriate bytes into key buffer
        for(int i = 0; i < buf_len; i++)
            key.buf[i] = key.buf[i] & buf_mask[i];
        auto search = hashmap->find(key);
        if (search == hashmap->end()) {
            hashmap->insert({key, 1});
        }
        else {
            search->second += 1;
        }
    }
    if (ferror(read_fd))
        cout << "Error:" << ferror(read_fd);
    delete[] buf_mask;
    fclose(read_fd);
}

ByteCont* read_samps_hit(unsigned long num_samps, int buf_len)  {
    FILE* read_fd = fopen("/dev/xillybus_read_32", "rb");
    if (read_fd==NULL) {fputs ("File error",stderr); exit (1);}
    int read_bytes = 0;
    ByteCont* key = new ByteCont(buf_len);
   
    for(unsigned long i = 0; i < num_samps; i++) {
        read_bytes += fread(key->buf, 1, buf_len, read_fd);
    }
    if (ferror(read_fd))
        cout << "Error:" << ferror(read_fd);
    
    fclose(read_fd);
    return key;
}


ByteCont* read_samps_hit(unsigned long num_samps, int buf_len, int num_visible, int* outbits, size_t outbits_len) {
    FILE* read_fd = fopen("/dev/xillybus_read_32", "rb");
    if (read_fd==NULL) {fputs ("File error",stderr); exit (1);}
    int read_bytes = 0;
    ByteCont* key = new ByteCont(buf_len);
   
    //Need extra () to ensure buf_mask is zeroed out
    char * buf_mask = new char[buf_len]();
    //Creates a buffer mask so that the hash only looks at the bits specified in the outbits array
    int bit_tot, byte_loc, bit_loc;
    for(size_t i = 0; i < outbits_len; i++) {
        bit_tot = num_visible - 1 - outbits[i];
        byte_loc = buf_len - (bit_tot / 8 + 1);
        //this should do conversion for xillybus byte indices
        byte_loc = 4*(byte_loc/4) + (3 - (byte_loc%4));
        bit_loc = bit_tot%8;
        buf_mask[byte_loc] = buf_mask[byte_loc] | (1 << bit_loc);  
    }
    
    for(unsigned long i = 0; i < num_samps; i++) {
        read_bytes += fread(key->buf, 1, buf_len, read_fd);
    }
    if (ferror(read_fd))
        cout << "Error:" << ferror(read_fd);
    //Masking appropriate bytes into key buffer
    for(int i = 0; i < buf_len; i++)
        key->buf[i] = key->buf[i] & buf_mask[i];
    
    delete[] buf_mask;
    fclose(read_fd);
    return key;
}

std::vector<ByteCont>* read_samps_raw(unsigned long num_samps, int buf_len) {
    FILE* read_fd = fopen("/dev/xillybus_read_32", "rb");
    if (read_fd==NULL) {fputs ("File error",stderr); exit (1);}
    int read_bytes = 0;
    std::vector<ByteCont>* outvec = new std::vector<ByteCont>(num_samps);
    for(unsigned long i = 0; i < num_samps; i++) {
        ByteCont* key = new ByteCont(buf_len);
        read_bytes += fread(key->buf, 1, buf_len, read_fd);
        (*outvec)[i] = *key;    
    }
    if (ferror(read_fd))
        cout << "Error:" << ferror(read_fd);
    //Masking appropriate bytes into key buffer
    
    fclose(read_fd);
    return outvec;

}



std::vector<int> inttovec(int* x) {
    return std::vector<int>(x, x+4);
}

void del_map(bytemap* hashmap) {
    delete hashmap;
}


void print_boolvec(vector<bool> bytemap) {
    cout << "[";
    for(bool b : bytemap) {
        cout << b << ",";
    }
    cout << "]";
}

void print_intvec(vector<int> intvec) {
    cout << "[";
    for(int v : intvec) 
        cout << v << ",";
    cout << "]";
}
/*
int main(int argc, char *argv[]) {
    bytemap hashmap;
    cout << "hello" << endl;
    int num_visible = 3;
    int outbits[] = {0, 1, 2};
    read_samps(&hashmap, 10000, 32, num_visible, outbits, 3);
    cout << "Hashmap" << endl;
    ByteCont first_val(hashmap.begin()->first);
    
    for(auto it = hashmap.begin(); it != hashmap.end(); ++it) {
        cout << convert(&(it->first), num_visible, outbits, 3);
        cout << ":" << it->second << endl;
    }
    

    cout << "Max: " << convert(argmax(&hashmap)) << endl;
    cout << "size of unsigned long long: " << sizeof(unsigned long long) << endl;

    cout << "size of int: " << sizeof(int) << endl;
    ByteCont test(32);
    for(int i = 0; i < 32; i++) 
        test.buf[i] = i;
    ByteHasher hasher;
    cout << hasher(test) << endl;
    cout << "We done" << endl;
}*/

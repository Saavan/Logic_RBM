CXX = nvcc
SWIG = /usr/bin/swig
SWIGFLAGS = -c++ -python 
CXXFLAGS = -c --compiler-options '-fPIC' -std=c++11 -O3 -arch=compute_70 #for debugging purposes
OUTFLAGS = -o
OUTNAME = rbm.out
SWIGINCLUDE = -I/usr/include/python3.6m -lpython3.6m
CUINCLUDE = -lcublas -lcurand
LDFLAGS = -shared
OBJS = rbm.o rbm_wrap.o
INS = utils.cpp rbm_kernels.cu bytecont.cpp
OUTS = _rbm.so 
SWG = rbm.i

$(OUTS): $(OBJS)
	$(CXX) $(CUINCLUDE) $(LDFLAGS) $^ -o $@

rbm.o: rbm.cu
	$(CXX) $(CXXFLAGS) $(CUINCLUDE) $<  -o $@

%.o: %.cu 
	$(CXX) $(CXXFLAGS) $(CUINCLUDE) $< -o $@
%.o: $.cpp
	$(CXX) $(CXXFLAGS) $(CUINCLUDE) $< -o $@

rbm_wrap.o: rbm_wrap.cxx
	$(CXX) $(CXXFLAGS) $(SWIGINCLUDE) $< -o $@ 
	
rbm_wrap.cxx: rbm.i
	$(SWIG) $(SWIGFLAGS) $(SWG)

profile: 
	$(CXX) rbm.cpp -o $(OUTNAME)
	valgrind --tool=callgrind ./$(OUTNAME)	

run: 
	$(CXX) rbm.cpp -o $(OUTNAME)

clean:
	rm $(OBJS) $(OUTS) %.o rbm_wrap.cxx rbm.py

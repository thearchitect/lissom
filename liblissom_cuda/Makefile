
FLAGS = -O2

CUDAPATH = /usr/local/cuda
CUDASDK = /usr/local/NVIDIA_CUDA_SDK
INC = -I$(CUDAPATH)/include -I$(CUDASDK)/common/inc
LIB_ = -L$(CUDAPATH)/lib -L$(CUDASDK)/lib -L$(CUDASDK)/common/lib

OBJS = obj/lissom.o obj/retina.o obj/lgn.o obj/CUDALISSOM.o obj/retinaLGN.o
LIB = liblissom.a


all: $(LIB)
	g++ -o lissom example/example.cpp -L. -llissom $(INC) $(LIB_) -lcuda -lcudart -lGL -lGLU -lcutil -I/usr/local/include/opencv -L/usr/local/lib -lcv -lcxcore -lcvaux -lhighgui $(FLAGS)


clean:
	rm -f obj/*.o
	rm -f liblissom.a
	rm -f lissom




$(LIB) : $(OBJS)
	ar rc liblissom.a obj/*.o


obj/CUDALISSOM.o : CUDA/cudaLissom.cu
	nvcc -c -o obj/CUDALISSOM.o CUDA/cudaLissom.cu $(FLAGS) $(INC)

obj/retinaLGN.o : CUDA/retinaLGN.cu
	nvcc -c -o obj/retinaLGN.o CUDA/retinaLGN.cu $(FLAGS) $(INC)

obj/lissom.o : C++/lissom.cpp
	g++ -c C++/lissom.cpp -o obj/lissom.o $(FLAGS) $(INC)

obj/retina.o : C++/retina.cpp
	g++ -c C++/retina.cpp -o obj/retina.o $(FLAGS) $(INC)

obj/lgn.o : C++/lgn.cpp
	g++ -c C++/lgn.cpp -o obj/lgn.o $(FLAGS) $(INC)



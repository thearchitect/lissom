
FLAGS = -O2
MPI = 
CC = g++


CUDAPATH = /usr/local/cuda
CUDASDK = /usr/local/NVIDIA_CUDA_SDK
INC = -I$(CUDAPATH)/include -I$(CUDASDK)/common/inc
LIB = -L$(CUDAPATH)/lib -L$(CUDASDK)/lib -L$(CUDASDK)/common/lib


OBJS = obj/GPUWorker.o obj/MPICUDALissom.o obj/MPIRetina.o


all: $(OBJS)
	ar rc libMPICUDALissom.a $(OBJS)

	$(CC) -o lissom example/example.cpp -L. -L../liblissom_cuda $(INC) $(LIB)  -lMPICUDALissom -llissom -lcuda -lcudart -lGL -lGLU -lcutil -lboost_thread -I/usr/local/include/opencv -L/usr/local/lib -lcv -lcxcore -lcvaux -lhighgui $(FLAGS) $(MPI)

mpi:
	MPI=-DWITH_MPI
	CC=mpicxx
	make all



clean:
	rm obj/*.o
	rm libMPICUDALissom.a
	rm lissom




obj/GPUWorker.o : src/GPUWorker.cc
	$(CC) -c -o obj/GPUWorker.o src/GPUWorker.cc $(FLAGS) $(INC) $(MPI)

obj/MPICUDALissom.o : src/MPICUDALissom.cpp
	$(CC) -c -o obj/MPICUDALissom.o src/MPICUDALissom.cpp $(FLAGS) $(INC) $(MPI)

obj/MPIRetina.o : src/MPIRetina.cpp
	$(CC) -c -o obj/MPIRetina.o src/MPIRetina.cpp $(FLAGS) $(INC) $(MPI)


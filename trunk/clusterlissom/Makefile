

FLAGS = -O2

OBJS = tmp/main.o tmp/config.o tmp/parser.o tmp/lissomwrapper.o tmp/afferentlayer.o


CUDAPATH = /usr/local/cuda
CUDASDK = /usr/local/NVIDIA_CUDA_SDK
INC = -I$(CUDAPATH)/include -I$(CUDASDK)/common/inc
LIB = -L$(CUDAPATH)/lib -L$(CUDASDK)/lib -L$(CUDASDK)/common/lib


all: $(OBJS)
	mpicxx -o clusterlissom $(OBJS) -L. $(INC) $(LIB) -llissom -lcuda -lcudart -lGL -lGLU -lcutil $(FLAGS)


clean:
	rm tmp/*.o
	rm clusterlissom



tmp/main.o : src/main.cpp
	mpicxx -c $(FLAGS) $(INC) -o tmp/main.o src/main.cpp

tmp/config.o : src/config.cpp
	mpicxx -c $(FLAGS) $(INC) -o tmp/config.o src/config.cpp

tmp/parser.o : src/parser.cpp
	mpicxx -c $(FLAGS) $(INC) -o tmp/parser.o src/parser.cpp

tmp/lissomwrapper.o : src/lissomwrapper.cpp
	mpicxx -c $(FLAGS) $(INC) -o tmp/lissomwrapper.o src/lissomwrapper.cpp

tmp/afferentlayer.o : src/afferentlayer.cpp
	mpicxx -c $(FLAGS) $(INC) -o tmp/afferentlayer.o src/afferentlayer.cpp



FLAGS = -O2

OBJS = obj/lissom.o obj/retina.o obj/lgn.o obj/CUDALISSOM.o obj/retinaLGN.o
LIB = liblissom.a


all: $(LIB)
	g++ -o lissom example/example.cpp -L. -llissom -lcuda -lcudart -lGL -lGLU -lcutil -I/usr/local/include/opencv -L/usr/local/lib -lcv -lcxcore -lcvaux -lhighgui $(FLAGS)


clean:
	rm -f obj/*.o
	rm -f liblissom.a
	rm -f lissom




$(LIB) : $(OBJS)
	ar rc liblissom.a obj/*.o


obj/CUDALISSOM.o : CUDA/cudaLissom.cu
	nvcc -c -o obj/CUDALISSOM.o CUDA/cudaLissom.cu $(FLAGS)

obj/retinaLGN.o : CUDA/retinaLGN.cu
	nvcc -c -o obj/retinaLGN.o CUDA/retinaLGN.cu $(FLAGS)

obj/lissom.o : C++/lissom.cpp
	g++ -c C++/lissom.cpp -o obj/lissom.o $(FLAGS)

obj/retina.o : C++/retina.cpp
	g++ -c C++/retina.cpp -o obj/retina.o $(FLAGS)

obj/lgn.o : C++/lgn.cpp
	g++ -c C++/lgn.cpp -o obj/lgn.o $(FLAGS)



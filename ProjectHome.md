![http://homepages.inf.ed.ac.uk/jbednar/images/000506_or_map_128MB.RF-LISSOM.anim.gif](http://homepages.inf.ed.ac.uk/jbednar/images/000506_or_map_128MB.RF-LISSOM.anim.gif)

LISSOM (http://homepages.inf.ed.ac.uk/jbednar/rflissom_small.html)
is a model of human neocortex (mainly modeled on visual cortex) at a
neural column level. The model was developed by Bednar, Choe,
Miikkulainen, and Sirosh, at the University of Texas. I made
different implementations of the model (with the help of
Prof. Bednar), with the aim of porting it to GPU hardware using CUDA
by nVidia. Now the library can run up to 9x faster than on modern
CPUs. The project also involves usage of multiGPU systems and GPU
clusters, to overcome the two big problems we face:

-lack of memory to store the network;

-computing power to simulate the model.


[Presentation of the project](http://lissom.googlecode.com/svn/wiki/Presentation.pdf) <br />
[Research Abstract of the project](http://lissom.googlecode.com/svn/wiki/abstract.pdf)

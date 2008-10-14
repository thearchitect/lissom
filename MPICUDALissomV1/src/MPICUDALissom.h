/*
    Copyright (C) 2008 Giacomo Spigler

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#ifndef __MPICUDALISSOM__H__
#define __MPICUDALISSOM__H__



#include "../../liblissom_cuda/C++/lissom.h"

#include "GPUWorker.h"


#define MAX_CPU_THREADS 4



class MPIRetina: public Layer {
public:
  int patternsperiteration;

  float *buf;

  int numGPUs;


  GPUWorker *g0;

  Retina *main;
  float *others[MAX_CPU_THREADS-1];
  char used[MAX_CPU_THREADS-1];


   MPIRetina(int w_, int h_, GPUWorker *context0, int patternsperiteration_=1);
  ~MPIRetina();

  //[ 1 ; numGPUs-1 ]
  void AllocateOnGPU(GPUWorker *g, int n);
  void CopyToGPU(GPUWorker *g, int n);

  void setPatternsPerIteration(int i) { if(i>=1) patternsperiteration=i; } //Don't forget to set it!

  void setinput(GPUWorker *g, int n, unsigned char *im, int widthstep=0);
  void randomGaussian(int centered=0, int number=0, float a2=56.25, float b2=2.25);
  void getoutput(unsigned char *im, int widthstep=0);

};




// TODO: At the moment it can only handle 1 afferent!
class MPICUDALissom {
public:
  int w, h;
  int hPerGPU;

  int numinputs;

  int numGPUs;


  MPIRetina **retinas;

  LISSOM *layers[MAX_CPU_THREADS];

  int offsets[MAX_CPU_THREADS];

  GPUWorker *g[MAX_CPU_THREADS];


  float *buf;



  MPICUDALissom(int w_, int h_, float rf_, GPUWorker *context0, MPIRetina *afferent, int numinputs_=1, int scaleAreaOrDensity=0, int weightsup=0, int weightsdown=0, float rE_=0.0, float rI_=0.0, float rEf_=1.13, float alphaA_=0.007, float alphaE_=0.00466167, float alphaI_=0.0330078, float gammaE_=0.9, float gammaI_=0.9, int settletime_=9, float lowerthr_=0.1, float upperthr_=0.65);
  ~MPICUDALissom();

  MPICUDALissom(char *file);
  void save(char *file);


  void setThresholds(float lower, float upper) { for(int i=0; i<numGPUs; i++) layers[i]->setThresholds(lower, upper); }
  void setGammas(float E, float I)  { for(int i=0; i<numGPUs; i++) layers[i]->setGammas(E, I); }
  void setAlphaA(float a) { for(int i=0; i<numGPUs; i++) layers[i]->setAlphaA(a); }
  void setAlphaE(float a)  { for(int i=0; i<numGPUs; i++) layers[i]->setAlphaE(a); }
  void setAlphaI(float a)  { for(int i=0; i<numGPUs; i++) layers[i]->setAlphaI(a); }
  void setRef(float r)  { for(int i=0; i<numGPUs; i++) layers[i]->setRef(r); }
  void setRe(float r) { for(int i=0; i<numGPUs; i++) layers[i]->setRe(r); }
  void setSettleTime(int t) { for(int i=0; i<numGPUs; i++) layers[i]->setSettleTime(t); }


  void getweight(unsigned char *im, int num, int x, int y, int widthstep=0);

  void ConnectAfferent(MPIRetina *afferent, int afferentnum=0);  //Be sure to check afferentnum!

  void getoutput(unsigned char *im, int widthstep=0);

  void normalizeweights();
  void FirstStep();
  void Step(int iters=-1);
  void AdjustWeights();

};
















#endif /* __MPICUDALISSOM__H__ */

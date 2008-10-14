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

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>


#include "GPUWorker.h"


#include <boost/bind.hpp>
using namespace boost;


#include "../../liblissom_cuda/CUDA/cudaLissom.h"
#include "../../liblissom_cuda/C++/lissom.h"


#include "MPICUDALissom.h"



cudaError_t NewRetina_(int w, int h, void **cuda) {
  *cuda=(void *)CUDANewRetina(w, h);

  return cudaSuccess;
}


cudaError_t RetinaRandomGaussian_(CUDALISSOM **cuda, int centered, float a2, float b2, int w, int h, int number) {
  CUDARetinaRandomGaussian(*cuda, centered, a2, b2, w, h, number, -1, -1, -1, 0.369f);

  return cudaSuccess;
}







MPIRetina::MPIRetina(int w_, int h_, GPUWorker *context0, int patternsperiteration_) {
  w=w_;
  h=h_;
  g0=context0;
  patternsperiteration=patternsperiteration_;

  type=LAYER_RETINA;

  buf=(float *)malloc(w*h*sizeof(float));

  g0->call(bind( NewRetina_, w, h, &cuda ));

  for(int i=0; i<MAX_CPU_THREADS-1; i++) used[i]=0;

}



void MPIRetina::AllocateOnGPU(GPUWorker *g, int n) {
  g->call(bind( cudaMalloc, (void **)&(others[n-1]), w*h*sizeof(float) ));

  used[n-1]=1;

}



MPIRetina::~MPIRetina() {
  free(buf);
//  g0->call(bind( CUDADeleteRetina, (CUDALISSOM *)cuda ));

  CUDALISSOM *cu=(CUDALISSOM *)cuda;
  g0->call(bind( cudaFree, cu->neurons ));
  free(cu);

  for(int i=0; i<MAX_CPU_THREADS-1; i++) {
    if(used[i]==1) cudaFree(others[i]);
  }

}



void MPIRetina::setinput(GPUWorker *g, int n, unsigned char *im, int widthstep) {
  if(widthstep==0) widthstep=w*sizeof(float);

  if(n==0) {
    CUDALISSOM *cu=(CUDALISSOM *)cuda;
    g0->call(bind( cudaMemcpy2D, cu->neurons, widthstep, im, w*sizeof(float), w*sizeof(float), h, cudaMemcpyHostToDevice ));

  } else {
    g->call(bind( cudaMemcpy2D, others[n-1], widthstep, im, w*sizeof(float), w*sizeof(float), h, cudaMemcpyHostToDevice ));
  }

}



void MPIRetina::getoutput(unsigned char *im, int widthstep) {
  if(widthstep==0) widthstep=w*sizeof(float);

  CUDALISSOM *cu=(CUDALISSOM *)cuda;
  g0->call(bind( cudaMemcpy2D, im, widthstep, cu->neurons, w*sizeof(float), w*sizeof(float), h, cudaMemcpyDeviceToHost ));

}



void MPIRetina::randomGaussian(int centered, int number, float a2, float b2) {
  if(number==0) number=patternsperiteration;

  g0->call(bind(  RetinaRandomGaussian_, (CUDALISSOM **)&cuda, centered, a2, b2, w, h, number ));

}



void MPIRetina::CopyToGPU(GPUWorker *g, int n) {
  CUDALISSOM *cu = (CUDALISSOM *)cuda;
  g0->call(bind( cudaMemcpy, buf, cu->neurons, w*h*sizeof(float), cudaMemcpyDeviceToHost ));

  g->call(bind( cudaMemcpy, others[n-1], buf, w*h*sizeof(float), cudaMemcpyHostToDevice ));

}















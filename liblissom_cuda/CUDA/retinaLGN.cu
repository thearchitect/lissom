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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <cutil.h>

#include "cudaLissom.h"


texture<float, 1, cudaReadModeElementType> texRetina;



__global__ void RetinaRandomGaussian( int w, int h, float *input, float a2, float b2, int xC, int yC, float s, float cost_, float thr=0.369f) {
  int size=w*h;

  for(unsigned int i=threadIdx.x+blockIdx.x*blockDim.x; i<size; i+=blockDim.x*gridDim.x) {
    int x=i % w;
    int y=(i/w);

    int j=x-xC;
    int k=y-yC;


    float res=expf( -(j*cost_ - k*s)*(j*cost_ - k*s)/a2 -(j*s + k*cost_)*(j*s + k*cost_)/b2 );

    if(res>=thr) {
      float current=input[y*w + x];
      input[y*w + x] = fmax(res, current);
    }

  }

}





__global__ void LGNRun(float *on, float *off, int w, int h, int inputw, int inputh, float ratioW, float ratioH, int rf) {
  int size=w*h;


  int rf2=2*rf+1;

  __shared__ float sum[THREADS];

  for(int i=blockIdx.x; i<size; i+=gridDim.x) {
    float temptemp=0.0f;

    int x=i % w;
    int y=(int)(i/w);


    for(int p=threadIdx.x; p<rf2*rf2; p+=blockDim.x) {
      int xx=(p % rf2) - 1 - rf;
      int yy=(int)(p/rf2) - 1 - rf;

      float prod=(float)(xx*xx+yy*yy);
//      float ww=expf(-prod/0.25f) / 1.0746f  -  expf(-prod/4.0f) / 12.56637f;
      float ww=expf(-prod/2.0f) / 6.283185f  -  expf(-prod/8.0f) / 25.13266f;


      xx += x + rf;
      yy += y + rf;


      float tmp = tex1Dfetch(texRetina, yy*ratioH*inputw + xx*ratioW );


      temptemp += ww * tmp;
    }

    __syncthreads();

    sum[threadIdx.x] = temptemp;

    __syncthreads();


    if(threadIdx.x==0) {
      #pragma unroll
      for(int p=1; p<THREADS; p++) {
        temptemp += sum[p];
      }


temptemp*=10.2f;

if(temptemp<=0.1f) {
  temptemp=0.0f;
} else if(temptemp>=0.65f) {
  temptemp=1.0f;
} else {
  temptemp=(temptemp-0.1f)/0.5f;
}

//      temptemp*=2.0f; //check this
      if(temptemp>0.0f) {
        //ON
        on[i] = temptemp;
      } else {
        //OFF
        off[i] = -temptemp;
      }

    }


  }




}










CUDALISSOM *CUDANewRetina(int w, int h) {
  CUDALISSOM *a=(CUDALISSOM *)malloc(sizeof(CUDALISSOM));

  cudaMalloc((void **)&(a->neurons), w*h*sizeof(float));

  return a;
}



CUDALISSOM *CUDANewLGN(int w, int h) {
  CUDALISSOM *a=(CUDALISSOM *)malloc(sizeof(CUDALISSOM));

  cudaMalloc( (void **)&(a->neurons), w*h*sizeof(float) ); //LGN ON
  cudaMalloc( (void **)&(a->temp), w*h*sizeof(float) ); //LGN OFF

  a->inputs_host=(float **)malloc(sizeof(float *)); //we only need 1 input channel

  return a;
}



void CUDADeleteRetina(CUDALISSOM *a) {
  cudaFree(a->neurons);
  free(a);
}



void CUDADeleteLGN(CUDALISSOM *a) {
  cudaFree(a->neurons);
  cudaFree(a->temp);
  free(a->inputs_host);
  free(a);
}



void CUDARetinaSetInput(CUDALISSOM *a, unsigned char *im, int widthstep, int w, int h) {
//TODO: check that it is working

  cudaMemcpy2D(a->neurons, w*sizeof(float), im, widthstep, w*sizeof(float), h, cudaMemcpyHostToDevice);
}



void CUDARetinaRandomGaussian(CUDALISSOM *a, int centered, float a2, float b2, int w, int h, int number, int x, int y, int angledeg, float thr) {
  #define BORDER 4

  #ifdef DEBUG
    unsigned int hTimer;
    cutCreateTimer(&hTimer);
    cutStartTimer(hTimer);
  #endif

  cudaMemset(a->neurons, 0, w*h*sizeof(float));

  for(int i=0; i<number; i++) {
    if(x==-1 || y==-1) {
      x=(int)((float)rand()/(float(RAND_MAX)+1.0)*(float)(w-2*BORDER))+BORDER;
      y=(int)((float)rand()/(float(RAND_MAX)+1.0)*(float)(h-2*BORDER))+BORDER;
    }
    int theta_;
    if(angledeg==-1) {
      theta_=(int)((float)rand()/(float(RAND_MAX)+1.0)*180.0);
      theta_=theta_/15;
      theta_=theta_*15;
    } else {
      theta_=angledeg;
    }

    float theta=(float)theta_/180.0 * 3.141592;

    if(centered==1) {
      x=w/2;
      y=h/2;
    }


    RetinaRandomGaussian<<<BLOCKS, THREADS>>>(w, h, a->neurons, a2, b2, x, y, sin(theta), cos(theta), thr); //Just generate Gaussian on the first afferent layer
    cudaThreadSynchronize();
  }

  #ifdef DEBUG
    cutStopTimer(hTimer);
    printf("Random Gaussian Generation Time: %fms\n", cutGetTimerValue(hTimer));
  #endif

}



void CUDAGetBuffer(float *buf, unsigned char *im, int widthstep, int w, int h) {
  cudaMemcpy2D(im, widthstep, buf, w*sizeof(float), w*sizeof(float), h, cudaMemcpyDeviceToHost);

}



void CUDALGNRun(CUDALISSOM *a, int w, int h, int inputw, int inputh, float ratioW, float ratioH, int rf) {
  #ifdef DEBUG
    unsigned int hTimer;
    cutCreateTimer(&hTimer);
    cutStartTimer(hTimer);
  #endif

  cudaMemset(a->neurons, 0, w*h*sizeof(float));
  cudaMemset(a->temp, 0, w*h*sizeof(float));

  cudaBindTexture(0, texRetina, a->inputs_host[0], inputw*inputh*sizeof(float));

  LGNRun<<<BLOCKS, THREADS>>>(a->neurons, a->temp, w, h, inputw, inputh, ratioW, ratioH, rf);
  cudaThreadSynchronize();

  cudaUnbindTexture(texRetina);

  #ifdef DEBUG
    cutStopTimer(hTimer);
    printf("LGN Processing Time: %fms\n", cutGetTimerValue(hTimer));
  #endif

}



void CUDASetAfferentLGN(CUDALISSOM *cuda, CUDALISSOM *aff, int startaff) {
  cuda->inputs_host[startaff] = aff->neurons; //lgn on
  cuda->inputs_host[startaff+1] = aff->temp; //lgn off
}
































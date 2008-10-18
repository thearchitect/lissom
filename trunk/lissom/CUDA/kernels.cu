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


#include "cudaLissom.h"


texture<float, 1, cudaReadModeElementType> texInput;
texture<CUDAWEIGHT, 1, cudaReadModeElementType> texWeights;
texture<CUDAWEIGHT, 1, cudaReadModeElementType> texWeightsI;



__device__ unsigned long random(unsigned long num) { //0..100000
  num = (num*267043 + 714) % 100000;
  return fabsf(num);
}



__global__ void InitGPUWeights(int w, int h, CUDAWEIGHT *weights, int *numreceptors, int type, float rf, float ratioW, float ratioH, float offset, unsigned long randseed, int *startindex, unsigned int inputw, int weightsup, int weightsdown, int offsety) {
  int size=w*h;

  int rfInt=(int)rf;
  int rf22=rfInt*rfInt;

  int rfext=2*rfInt+1;

  int rf2=rfext*rfext;

  int count;

  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<size; i+=gridDim.x*blockDim.x) {
    int x=(int)(i % w);
    int y=(int)(i/w);

    count=0;

    for(int p=0; p<rf2; p++) {
      int j=(int)( p % rfext ) - rfInt;
      int k=(int)( p/rfext ) - rfInt;


      if(j*j + k*k <= rf22) {

        if( type==PROJECTION_AFFERENT || (type!=PROJECTION_AFFERENT && (x+j>=0 && x+j<w && (y+k>=0 || weightsup==1) && (y+k<h || weightsdown==1) ) ) ) {
          randseed=random(randseed);

          float val=(float)randseed / 100000.0f;
          val=(val+0.05)/1.05;


          CUDAWEIGHT tmp;
          if(type==PROJECTION_AFFERENT) {
            int xxx=((float)x* ratioW + offset + j); //+offset MPICUDALissom
            int yyy=((float)(y+offsety)* ratioH + offset + k); //+offset  //NOTE: offsety is used by MPICUDALissom, in which case ratioh's used, too
            tmp.x=yyy*inputw + xxx;
          } else {
            int xxx=(x + j);
            int yyy=(y + k   + offsety);
            tmp.x=yyy*w + xxx;
          }
          tmp.y=val;

          weights[startindex[i] + count] = tmp;


          count++;


        }
      }


    }

  }

}



__global__ void NormalizeWeights(int w, int h, CUDAWEIGHT *weights, int *numreceptors, int *startindex, int afferent, float *temp) {
  int size=w*h;


  __shared__ int numR;
  __shared__ int start;
  __shared__ float sum[THREADS];

  __shared__ float current;

  for(int i=blockIdx.x; i<size; i+=gridDim.x) {
    if(threadIdx.x==0) {
      numR=numreceptors[i];
      start=startindex[i];

      if(afferent==1) current=temp[i];
    }

    __syncthreads();

    float temptemp = 0.0f;

    for(int p=threadIdx.x; p<numR; p+=blockDim.x) {
      temptemp += tex1Dfetch(texWeights, start + p).y; //weights[start + p].y;
    }
    __syncthreads();
    sum[threadIdx.x]=temptemp;
    __syncthreads();


   if(threadIdx.x==0) {
      #pragma unroll
      for(int p=1; p<THREADS; p++) {
        sum[0] += sum[p];
      }

      if(afferent==1) temp[i] = sum[0]+current;
    }
    __syncthreads();


    if(afferent==0) {
      float s=sum[0];

      for(int p=threadIdx.x; p<numR; p+=blockDim.x) {
  //      weights[start + p].y /= s;
        float ttt = tex1Dfetch(texWeights, start + p).y;
        weights[start + p].y = ttt/s;
      }

    }


  }


}



__global__ void NormalizeWeightsAfferent(int w, int h, CUDAWEIGHT *weights, int *numreceptors, int *startindex, float *temp) {
  int size=w*h;


  __shared__ int numR;
  __shared__ int start;

  __shared__ float current;

  for(int i=blockIdx.x; i<size; i+=gridDim.x) {
    if(threadIdx.x==0) {
      numR=numreceptors[i];
      start=startindex[i];

      current = temp[i];
    }

    __syncthreads();

    for(int p=threadIdx.x; p<numR; p+=blockDim.x) {
      float ttt = tex1Dfetch(texWeights, start + p).y; //weights[start + p].y;
      weights[start + p].y = ttt/current;
    }

  }


}



//TODO: ATTENTION: only the first layer is used!
__global__ void RandomGaussian(unsigned int inputWGPU, int inputw, int inputh, float **inputs, float a2, float b2, int xC, int yC, float s, float cost_) {
  int size=inputw*inputh;

  for(unsigned int i=threadIdx.x+blockIdx.x*blockDim.x; i<size; i+=blockDim.x*gridDim.x) {
    int x=i % inputw;
    int y=(i/inputh);

    int j=x-xC;
    int k=y-yC;


    float res=expf( -(j*cost_ - k*s)*(j*cost_ - k*s)/a2 -(j*s + k*cost_)*(j*s + k*cost_)/b2 );

    if(res>0.369f) {
      float current=inputs[0][y*inputWGPU + x];
      inputs[0][y*inputWGPU + x] = fmax(res, current);
    }

  }

}




__global__ void FirstStep(CUDAWEIGHT *weights, int *numreceptors, float *input, float *temp, int w, int h, unsigned int inputWGPU, int *startindex) {
  int size=w*h;

  __shared__ int numR;
  __shared__ int start;
  __shared__ float sum[THREADS];

  for(int i=blockIdx.x; i<size; i+=gridDim.x) {
    if(threadIdx.x==0) {
      numR=numreceptors[i];
      start=startindex[i];
    }

    __syncthreads();


    float temptemp=0.0f;

    if(threadIdx.x==0) {
      temptemp=temp[i];
    }


    for(int p=threadIdx.x; p<numR; p+=blockDim.x) {
      CUDAWEIGHT tmp = tex1Dfetch(texWeights, start + p); //weights[start + p];

      float inp = tex1Dfetch(texInput, tmp.x); //input[tmp.y*inputWGPU + tmp.x];
      temptemp += tmp.y * inp;
    }

    __syncthreads();

    sum[threadIdx.x] = temptemp;

    __syncthreads();


    if(threadIdx.x==0) {
      #pragma unroll
      for(int p=1; p<THREADS; p++) {
        temptemp += sum[p];
      }



      if(temptemp>0.0f) temp[i] = temptemp;


    }


  }

}



__global__ void ActivationFunction(float *input, float *output, int w, int h, float lowerthr, float upperthr) {
  int size=w*h;

  float bmt = upperthr - lowerthr;

  for(int i=blockIdx.x*blockDim.x + threadIdx.x; i<size; i+=blockDim.x*gridDim.x) {
    float val = input[i];

    if(val>0.0f) {
      if(val<=lowerthr) {
        val = 0.0f;
      } else if(val>=upperthr) {
        val = 1.0f;
      } else {
        val = (val-lowerthr) / bmt;
      }

      output[i] = val;

    }

  }

}



__global__ void Step(int *numreceptors, float *temp, float *neurons, int w, int h, float gamma, int *startindex,   int *numreceptorsI, float gammaI, int *startindexI) {
  int size=w*h;

  __shared__ int numR, numRI;
  __shared__ int start, startI;
  __shared__ float sum[THREADS];
  __shared__ float sumI[THREADS];

  for(int i=blockIdx.x; i<size; i+=gridDim.x) {
    if(threadIdx.x==0) {
      numR=numreceptors[i];
      start=startindex[i];

      numRI=numreceptorsI[i];
      startI=startindexI[i];
    }

    __syncthreads();


    float temptemp=0.0f;
    float temptempI=0.0f;


    for(int p=threadIdx.x; p<numR; p+=blockDim.x) {
      CUDAWEIGHT tmp = tex1Dfetch(texWeights, start + p); //weights[start + p];

      if(tmp.x!=65000.0f) {
        float inp = tex1Dfetch(texInput, tmp.x); //neurons[tmp.y*w + tmp.x];
        temptemp = temptemp + tmp.y * inp;
      }

    }
    for(int p=threadIdx.x; p<numRI; p+=blockDim.x) {
      CUDAWEIGHT tmp = tex1Dfetch(texWeightsI, startI + p); //weights[start + p];

      if(tmp.x!=65000.0f) {
        float inp = tex1Dfetch(texInput, tmp.x); //neurons[tmp.y*w + tmp.x];
        temptempI = temptempI + tmp.y * inp;
      }

    }

    __syncthreads();

    sum[threadIdx.x] = temptemp;
    sumI[threadIdx.x] = temptempI;

    __syncthreads();


    if(threadIdx.x==0) {
      #pragma unroll
      for(int p=1; p<THREADS; p++) {
        temptemp += sum[p];
        temptempI += sumI[p];
      }

      temptemp *= gamma;
      temptempI *= gammaI;

      temp[i] += temptemp + temptempI;

    }


  }


}



__global__ void SetRe(int w, int h, CUDAWEIGHT *weights, int *numreceptors, float r, int *startindex, int offsety) {
  int size=w*h;

  int rfInt=(int)r;
  int rf22=rfInt*rfInt;

  __shared__ int numR;
  __shared__ int start;

  for(int i=blockIdx.x; i<size; i+=gridDim.x) {
    int x=i % w;
    int y=(int)(i / w);

    if(threadIdx.x==0) {
      numR=numreceptors[i];
      start=startindex[i];
    }

    __syncthreads();

    for(int p=threadIdx.x; p<numR; p+=blockDim.x) {
      CUDAWEIGHT tmp = tex1Dfetch(texWeights, start + p); //weights[start + p];
//TODO: ATTENTION: code the learning function to avoid 65000s
      int xxx=x - (unsigned int)tmp.x % w;
      int yyy=y - ( (int)((unsigned int)tmp.x / w) - offsety );
      if(xxx*xxx + yyy*yyy > rf22) {
        tmp.x=65000.0f;
        tmp.y=0.0f;
        weights[start + p] = tmp;
      }

    }
  }

}



__global__ void AdjustWeights(CUDAWEIGHT *weights, int *numreceptors, float *input, float *neurons, int inputW, int w, int h, float alpha, int *startindex, int afferent, float *temp, int offsety) {
  int size=w*h;

  __shared__ int numR;
  __shared__ int start;
  __shared__ float neur_;

  __shared__ float sum[THREADS];

  __shared__ float current;

  for(int i=blockIdx.x; i<size; i+=gridDim.x) {
    if(threadIdx.x==0) {
      numR=numreceptors[i];
      neur_=neurons[i+w*offsety];
      start=startindex[i];

      if(afferent==1) current=temp[i];
    }

    __syncthreads();

    float neuronval=neur_;


    //Normalize Weights
    float tempsum = 0.0f;


    for(int p=threadIdx.x; p<numR; p+=blockDim.x) {
      CUDAWEIGHT tmp = tex1Dfetch(texWeights, start + p); //weights[start + p];

      if(tmp.x!=65000.0f) {
        float tmp_ = tmp.y + alpha*neuronval * tex1Dfetch(texInput, tmp.x);

        tempsum += tmp_;
      }

    }

    sum[threadIdx.x] = tempsum;
    __syncthreads();


    if(threadIdx.x==0) {
      #pragma unroll
      for(int p=1; p<THREADS; p++) {
        tempsum += sum[p];
      }

      sum[0] = tempsum;
      if(afferent==1) temp[i] = sum[0] + current;
    }
    __syncthreads();


    if(afferent==0) {
      tempsum = sum[0];

      for(int p=threadIdx.x; p<numR; p+=blockDim.x) {
        CUDAWEIGHT tmp = tex1Dfetch(texWeights, start + p); //weights[start + p];

        if(tmp.x!=65000.0f) {
          float tmp_ = tmp.y + alpha*neuronval * tex1Dfetch(texInput, tmp.x);
          weights[start + p].y = tmp_ / tempsum; //input[tmp.y*inputW + tmp.x];
        }

      }

    }


  }


}



__global__ void AdjustWeightsAfferent(CUDAWEIGHT *weights, int *numreceptors, float *input, float *neurons, int inputW, int w, int h, float alpha, int *startindex, float *temp, int offsety) {
  int size=w*h;

  __shared__ int numR;
  __shared__ int start;
  __shared__ float neur_;

  __shared__ float current;

  for(int i=blockIdx.x; i<size; i+=gridDim.x) {
    if(threadIdx.x==0) {
      numR=numreceptors[i];
      neur_=neurons[i+w*offsety];
      start=startindex[i];

      current = temp[i];
    }

    __syncthreads();

    float neuronval=neur_;


    for(int p=threadIdx.x; p<numR; p+=blockDim.x) {
      CUDAWEIGHT tmp = tex1Dfetch(texWeights, start + p); //weights[start + p];

      if(tmp.x!=65000.0f) {
        float tmp_ = tmp.y + alpha*neuronval * tex1Dfetch(texInput, tmp.x);
        weights[start + p].y = tmp_ / current; //input[tmp.y*inputW + tmp.x];
      }

    }


  }


}




























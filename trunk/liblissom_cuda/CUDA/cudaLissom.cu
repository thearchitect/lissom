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

#include "kernels.cu"

#include "cudaLissom.h"


#define FABS(x) ( x>=0 ? x : -x )


#define writeint(a__, fp__) fwrite(&a__, sizeof(int), 1, fp__);
#define writefloat(a__, fp__) fwrite(&a__, sizeof(float), 1, fp__)
#define writelong(a__, fp__) fwrite(&a__, sizeof(long), 1, fp__);

#define readint(a__, fp__) fread(&a__, sizeof(int), 1, fp__);
#define readfloat(a__, fp__) fread(&a__, sizeof(float), 1, fp__)
#define readlong(a__, fp__) fread(&a__, sizeof(long), 1, fp__);



extern "C" {
//#define DEBUG




CUDALISSOM *NewCUDALISSOM(int w, int h, int inputw, int inputh, unsigned int *inputWGPU, int numinputs, float rA, float rE, float rI, float alphaA, float alphaE, float alphaI, float ratioW, float ratioH, float offset, float gammaE, float gammaI, int weightsup, int weightsdown, int donotinit, int offsety, float offsetyAff, int realh) {
  srand((unsigned)(time(0)));

  if(realh==0) realh=h;


  CUDALISSOM *a=(CUDALISSOM *)malloc(sizeof(CUDALISSOM));


  a->inputs_host=(float **)malloc(numinputs*sizeof(float *));
//  cudaMalloc((void **)&(a->inputs), numinputs*sizeof(float *));
//  for(int i=0; i<numinputs; i++) {
//    cudaMallocPitch( (void **)&(a->inputs_host[i]), inputWGPU, inputw*sizeof(float), inputh );
//  }

//  *inputWGPU /= sizeof(float);
  *inputWGPU = inputw; //TODO: VERY IMPORTANT FOR WEIGHTS INITIALIZATION (coords)!!!

//  cudaMemcpy(a->inputs, a->inputs_host, numinputs*sizeof(float *), cudaMemcpyHostToDevice);


  cudaMalloc((void **)&(a->neurons), w*h*sizeof(float));
  cudaMalloc((void **)&(a->temp), w*h*sizeof(float));
  cudaMalloc((void **)&(a->firstactivation), w*h*sizeof(float));


  a->projections=(CUDAPROJECTION **)malloc((2+numinputs) *sizeof(CUDAPROJECTION *));


  if(donotinit==0) {

    for(int i=0; i<numinputs; i++) {
      //Afferent
      a->projections[i+2]=NewCUDAPROJECTION(w, h, PROJECTION_AFFERENT, rA, alphaA, ratioW, ratioH, offset, 0.0, 0, 0, i);
    }
    //Excitatory
    a->projections[0]=NewCUDAPROJECTION(w, h, PROJECTION_EXCITATORY, rE, alphaE, ratioW, ratioH, offset, gammaE, weightsup, weightsdown);
    //Inhibitory
    a->projections[1]=NewCUDAPROJECTION(w, h, PROJECTION_INHIBITORY, rI, alphaI, ratioW, ratioH, offset, gammaI, weightsup, weightsdown);


    //Init GPU weights
    #ifdef DEBUG
      unsigned int hTimer;
      cutCreateTimer(&hTimer);
      cutStartTimer(hTimer);
    #endif

if(offsetyAff>=0.001) {
  ratioH=offsetyAff;
}



    for(int i=0; i<numinputs+2; i++) {
      long randseed=(long)(FABS((float)rand()/(float(RAND_MAX)+1.0))*100000.0);

      InitGPUWeights<<<BLOCKS, THREADS>>>(w, h, a->projections[i]->weights, a->projections[i]->numreceptors, a->projections[i]->type, a->projections[i]->rf, ratioW, ratioH, offset, randseed, a->projections[i]->startindex, *inputWGPU, weightsup, weightsdown, offsety, realh);
      cudaThreadSynchronize();
    }


    #ifdef DEBUG
      cutStopTimer(hTimer);
      printf("Weights Initialization time: %fms\n", cutGetTimerValue(hTimer));
    #endif


    CUDANormalizeWeights(a, numinputs, w, h);


  }


  return a;
}



void CUDANormalizeWeights(CUDALISSOM *a, int numinputs, int w, int h) {
  //Normalize GPU weights
  #ifdef DEBUG
    unsigned int hTimer;
    cutCreateTimer(&hTimer);
    cutStartTimer(hTimer);
  #endif

  cudaMemset(a->temp, 0, w*h*sizeof(float));

  for(int i=2; i<2+numinputs; i++) {
    cudaBindTexture(0, texWeights, a->projections[i]->weights, a->projections[i]->weightssize*sizeof(CUDAWEIGHT));

    NormalizeWeights<<<BLOCKS, THREADS>>>(w, h, a->projections[i]->weights, a->projections[i]->numreceptors, a->projections[i]->startindex, 1, a->temp);
    cudaThreadSynchronize();

    cudaUnbindTexture(texWeights);
  }


  for(int i=2; i<2+numinputs; i++) {
    cudaBindTexture(0, texWeights, a->projections[i]->weights, a->projections[i]->weightssize*sizeof(CUDAWEIGHT));

    NormalizeWeightsAfferent<<<BLOCKS, THREADS>>>(w, h, a->projections[i]->weights, a->projections[i]->numreceptors, a->projections[i]->startindex, a->temp);
    cudaThreadSynchronize();

    cudaUnbindTexture(texWeights);
  }


  for(int i=0; i<2; i++) {
    cudaBindTexture(0, texWeights, a->projections[i]->weights, a->projections[i]->weightssize*sizeof(CUDAWEIGHT));

    NormalizeWeights<<<BLOCKS, THREADS>>>(w, h, a->projections[i]->weights, a->projections[i]->numreceptors, a->projections[i]->startindex, 0, 0);
    cudaThreadSynchronize();

    cudaUnbindTexture(texWeights);
  }


  #ifdef DEBUG
    cutStopTimer(hTimer);
    printf("Weights normalization time: %fms\n", cutGetTimerValue(hTimer));
  #endif


}



void CUDASetGammas(CUDALISSOM *a, float E, float I) {
  a->projections[0]->gamma=E;
  a->projections[1]->gamma=I;
}



void CUDASetAlphaA(CUDALISSOM *b, float a, int numinputs) {
  for(int i=2; i<2+numinputs; i++) {
    b->projections[i]->alpha = a;
  }
}



void CUDASetAlphaE(CUDALISSOM *b, float a) {
  b->projections[0]->alpha = a;
}



void CUDASetAlphaI(CUDALISSOM *b, float a) {
  b->projections[1]->alpha = a;
}



void CUDASetRe(CUDALISSOM *a, float r, int w, int h, int offsety) {

  #ifdef DEBUG
    unsigned int hTimer;
    cutCreateTimer(&hTimer);
    cutStartTimer(hTimer);
  #endif

  a->projections[0]->rf = r;

  cudaBindTexture(0, texWeights, a->projections[0]->weights, a->projections[0]->weightssize*sizeof(CUDAWEIGHT));

  SetRe<<<BLOCKS, THREADS>>>(w, h, a->projections[0]->weights, a->projections[0]->numreceptors, r, a->projections[0]->startindex, offsety);
  cudaThreadSynchronize();

  cudaUnbindTexture(texWeights);

  #ifdef DEBUG
    cutStopTimer(hTimer);
    printf("Excitatory radius change time: %fms\n", cutGetTimerValue(hTimer));
  #endif

}



void DeleteCUDALISSOM(CUDALISSOM *a, int numinputs, int w, int h) {
  cudaFree(a->firstactivation);
  cudaFree(a->temp);
  cudaFree(a->neurons);


//  for(int i=0; i<numinputs; i++) {
//    cudaFree(a->inputs_host[i]);
//  }

//  cudaFree(a->inputs);
  free(a->inputs_host);


  for(int i=0; i<numinputs+2; i++) {
    DeleteCUDAPROJECTION(a->projections[i], w, h);
  }

  free(a->projections);


  free(a);
}



CUDAPROJECTION *NewCUDAPROJECTION(int w, int h, int type, float rf, float alpha, float ratioW, float ratioH, float offset, float gamma, int weightsup, int weightsdown, int afferentnum) {
  CUDAPROJECTION *a=(CUDAPROJECTION *)malloc(sizeof(CUDAPROJECTION));

  a->type=type;
  a->rf=rf;
  a->alpha=alpha;
  a->afferentnum=afferentnum;
  a->gamma=gamma;


  a->numreceptors_host=(int *)malloc(w*h *sizeof(int));
  a->startindex_host=(long long *)malloc(w*h*sizeof(long long));

  cudaMalloc( (void **)&(a->numreceptors), w*h *sizeof(int) );
  cudaMalloc( (void **)&(a->startindex), w*h *sizeof(long long) );

  long long weightssize=0;

  int rfInt=(int)rf;
  int rf22=rfInt*rfInt;

  //Fill numreceptors_host, init weights_host (allocate the GPU arrays)
  for(int x=0; x<w; x++) {
    for(int y=0; y<h; y++) {
      //a->weights_host[y*w+x]  * ALLOCATE: [connection *CALCULATE 'numconnections'* ] *

      int numconnections=0;

      for(int i=-rfInt; i<=rfInt; i++) {
        for(int j=-rfInt; j<=rfInt; j++) {
          if(i*i+j*j <= rf22) {
            if( type==PROJECTION_AFFERENT || (type!=PROJECTION_AFFERENT && (x+i>=0 && x+i<w && (y+j>=0 || weightsup==1) && (y+j<h || weightsdown==1) ) ) ) {
              numconnections++;
            }

          }

        }
      }


      int Windex=y*w + x;
      a->numreceptors_host[Windex] = numconnections;

      a->startindex_host[Windex] = weightssize;

      weightssize += numconnections;
    }
  }


  cudaMalloc( (void **)&(a->weights), weightssize*sizeof(CUDAWEIGHT) );


  a->weightssize=weightssize;


  //Copy numreceptors_host and weights_host to GPU
  cudaMemcpy( a->numreceptors, a->numreceptors_host, w*h*sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy( a->startindex, a->startindex_host, w*h*sizeof(long long), cudaMemcpyHostToDevice );


  return a;
}



void DeleteCUDAPROJECTION(CUDAPROJECTION *a, int w, int h) {
  free(a->numreceptors_host);
  cudaFree(a->numreceptors);

  cudaFree(a->weights);


  free(a);
}


/*
void CUDASetInput(CUDALISSOM *a, unsigned char *im, int widthstep, int inputnum, unsigned int inputWGPU, int inputw, int inputh) {
  cudaMemcpy2D(a->inputs_host[inputnum], widthstep, im, inputWGPU*sizeof(float), inputw*sizeof(float), inputh, cudaMemcpyHostToDevice);
}



void CUDAGetInput(CUDALISSOM *a, unsigned char *im, int widthstep, int inputnum, unsigned int inputWGPU, int inputw, int inputh) {
  cudaMemcpy2D(im, widthstep, a->inputs_host[inputnum], inputWGPU*sizeof(float), inputw*sizeof(float), inputh, cudaMemcpyDeviceToHost);
}
*/


void CUDAGetWeight(CUDALISSOM *a, unsigned char *im, int widthstep, int num, int x, int y, int w, int h, int inputw, int inputh, int offsety, float ratio) { //0=exc, 1=inhib, 2..n=afferent

  int Windex=(y-offsety)*w + x;
  int start=a->projections[num]->startindex_host[Windex];
  int nn=a->projections[num]->numreceptors_host[Windex];
  CUDAWEIGHT *we=(CUDAWEIGHT *)malloc(nn *sizeof(CUDAWEIGHT));
//weights[windex] -> we. then render them to image


  cudaMemcpy( we, a->projections[num]->weights+start, nn*sizeof(CUDAWEIGHT), cudaMemcpyDeviceToHost );


  for(int i=0; i<nn; i++) {
    int xxx, yyy;
    int xx, yy;
    if(a->projections[num]->type==PROJECTION_AFFERENT) {
      xxx=(int)we[i].x % inputw;
      yyy=(int)we[i].x / inputw;

      yyy=yyy-(float)y*ratio;

      xx=xxx-x + inputw/2;
      yy=yyy   + inputh/2;
    } else {
      xxx=(int)we[i].x % w;
      yyy=(int)we[i].x / w;

      xx=xxx-x + inputw/2;
      yy=yyy-y + inputh/2;
    }


    if(xx>=0 && xx<inputw && yy>=0 && yy<inputh) {
      ((float *)im)[yy*inputw + xx] = we[i].y*3.0 *10.0;
    }

  }

  free(we);

}



void CUDAGetOutput(CUDALISSOM *a, unsigned char *im, int widthstep, int w, int h) {
  cudaMemcpy2D(im, widthstep, a->neurons, w*sizeof(float), w*sizeof(float), h, cudaMemcpyDeviceToHost);
}


/*
void CUDARandomGaussian(CUDALISSOM *a, int centered, float a2, float b2, unsigned int inputWGPU, int inputw, int inputh, int number) {
  #define BORDER 4

  #ifdef DEBUG
    unsigned int hTimer;
    cutCreateTimer(&hTimer);
    cutStartTimer(hTimer);
  #endif

  cudaMemset(a->inputs_host[0], 0, inputWGPU*inputh*sizeof(float));
//  cudaMemset2D(a->inputs_host[0], inputWGPU, 0, inputw*sizeof(float), inputh);  //ATTENTION: host[0]!!!! only the first layer

  for(int i=0; i<number; i++) {
    int x=(int)((float)rand()/(float(RAND_MAX)+1.0)*(float)(inputw-2*BORDER))+BORDER;
    int y=(int)((float)rand()/(float(RAND_MAX)+1.0)*(float)(inputh-2*BORDER))+BORDER;
    int theta_=(int)((float)rand()/(float(RAND_MAX)+1.0)*180.0);
    theta_=theta_/15;
    theta_=theta_*15;

    float theta=(float)theta_/180.0 * 3.141592;

    if(centered==1) {
      x=inputw/2;
      y=inputh/2;
    }


    RandomGaussian<<<BLOCKS, THREADS>>>(inputWGPU, inputw, inputh, a->inputs, a2, b2, x, y, sin(theta), cos(theta)); //Just generate Gaussian on the first afferent layer
    cudaThreadSynchronize();
  }

  #ifdef DEBUG
    cutStopTimer(hTimer);
    printf("Random Gaussian Generation time: %fms\n", cutGetTimerValue(hTimer));
  #endif

}
*/


void CUDAFirstStep(CUDALISSOM *a, int w, int h, float lowerthr, float upperthr, int numinputs, unsigned int inputWGPU, int inputh) {

  #ifdef DEBUG
    unsigned int hTimer;
    cutCreateTimer(&hTimer);
    cutStartTimer(hTimer);
  #endif

  cudaMemset(a->temp, 0, w*h*sizeof(float));

  for(int i=0; i<numinputs; i++) {
    cudaBindTexture(0, texInput, a->inputs_host[a->projections[2+i]->afferentnum], inputWGPU*inputh*sizeof(float)); //*sizeof(float)?

    cudaBindTexture(0, texWeights, a->projections[2+i]->weights, a->projections[2+i]->weightssize*sizeof(CUDAWEIGHT));

    FirstStep<<<BLOCKS, THREADS>>>(a->projections[2+i]->weights, a->projections[2+i]->numreceptors, a->inputs_host[a->projections[2+i]->afferentnum], a->temp, w, h, inputWGPU, a->projections[2+i]->startindex);
    cudaThreadSynchronize();

    cudaUnbindTexture(texWeights);

    cudaUnbindTexture(texInput);
  }

  cudaMemcpy( a->firstactivation, a->temp, w*h*sizeof(float), cudaMemcpyDeviceToDevice );

  cudaMemset(a->neurons, 0, w*h*sizeof(float));

  ActivationFunction<<<BLOCKS, THREADS>>>(a->firstactivation, a->neurons, w, h, lowerthr, upperthr);
  cudaThreadSynchronize();

  #ifdef DEBUG
    cutStopTimer(hTimer);
    printf("First step time: %fms\n", cutGetTimerValue(hTimer));
  #endif

}



void CUDAStep(CUDALISSOM *a, int w, int h, float lowerthr, float upperthr, int realh) {

  #ifdef DEBUG
    unsigned int hTimer;
    cutCreateTimer(&hTimer);
    cutStartTimer(hTimer);
  #endif


  cudaMemcpy( a->temp, a->firstactivation, w*h*sizeof(float), cudaMemcpyDeviceToDevice );


  int maxh=h;
  if(realh!=0) maxh=realh;
  cudaBindTexture(0, texInput, a->neurons, w*maxh*sizeof(float)); //*sizeof(float)?

  cudaBindTexture(0, texWeights, a->projections[0]->weights,  a->projections[0]->weightssize*sizeof(CUDAWEIGHT));
  cudaBindTexture(0, texWeightsI, a->projections[1]->weights, a->projections[1]->weightssize*sizeof(CUDAWEIGHT));

  Step<<<BLOCKS, THREADS>>>(a->projections[0]->numreceptors, a->temp, a->neurons, w, h, a->projections[0]->gamma, a->projections[0]->startindex,    a->projections[1]->numreceptors, a->projections[1]->gamma, a->projections[1]->startindex);
  cudaThreadSynchronize();

  cudaUnbindTexture(texWeights);
  cudaUnbindTexture(texWeightsI);

  cudaUnbindTexture(texInput);

  cudaMemset(a->neurons, 0, w*h*sizeof(float));

  ActivationFunction<<<BLOCKS, THREADS>>>(a->temp, a->neurons, w, h, lowerthr, upperthr);
  cudaThreadSynchronize();

  #ifdef DEBUG
    cutStopTimer(hTimer);
    printf("Step time: %fms\n", cutGetTimerValue(hTimer));
  #endif

}



void CUDAAdjustWeights(CUDALISSOM *a, int w, int h, int inputWGPU, int inputh, int numinputs, int realh, int offsety) {
  if(realh==0) realh=h;

  #ifdef DEBUG
    unsigned int hTimer;
    cutCreateTimer(&hTimer);
    cutStartTimer(hTimer);
  #endif

  cudaMemset(a->temp, 0, w*realh*sizeof(float));

  for(int i=0; i<numinputs; i++) {
    float *input=a->inputs_host[a->projections[i+2]->afferentnum];
    int inputW=inputWGPU;

    cudaBindTexture(0, texInput, input, inputW*inputh*sizeof(float)); //*sizeof(float)?

    cudaBindTexture(0, texWeights, a->projections[i+2]->weights, a->projections[2+i]->weightssize*sizeof(CUDAWEIGHT));

    AdjustWeights<<<BLOCKS, THREADS>>>(a->projections[i+2]->weights, a->projections[i+2]->numreceptors, input, a->neurons, inputW, w, h, a->projections[i+2]->alpha, a->projections[i+2]->startindex, 1, a->temp, offsety);
    cudaThreadSynchronize();


    cudaUnbindTexture(texWeights);

    cudaUnbindTexture(texInput);
  }


  //Normalize afferent weights
  for(int i=0; i<numinputs; i++) {
    float *input=a->inputs_host[a->projections[i+2]->afferentnum];
    int inputW=inputWGPU;

    cudaBindTexture(0, texInput, input, inputW*inputh*sizeof(float)); //*sizeof(float)?

    cudaBindTexture(0, texWeights, a->projections[i+2]->weights, a->projections[i+2]->weightssize*sizeof(CUDAWEIGHT));

    AdjustWeightsAfferent<<<BLOCKS, THREADS>>>(a->projections[i+2]->weights, a->projections[i+2]->numreceptors, input, a->neurons, inputW, w, h, a->projections[i+2]->alpha, a->projections[i+2]->startindex, a->temp, offsety);
    cudaThreadSynchronize();

    cudaUnbindTexture(texWeights);

    cudaUnbindTexture(texInput);
  }



  for(int i=0; i<2; i++) {
    float *input=a->neurons;
    int inputW=w;

    cudaBindTexture(0, texInput, input, w*realh*sizeof(float)); //*sizeof(float)?

    cudaBindTexture(0, texWeights, a->projections[i]->weights, a->projections[i]->weightssize*sizeof(CUDAWEIGHT));

    AdjustWeights<<<BLOCKS, THREADS>>>(a->projections[i]->weights, a->projections[i]->numreceptors, input, a->neurons, inputW, w, h, a->projections[i]->alpha, a->projections[i]->startindex, 0, 0, offsety);
    cudaThreadSynchronize();

    cudaUnbindTexture(texWeights);

    cudaUnbindTexture(texInput);
  }


  #ifdef DEBUG
    cutStopTimer(hTimer);
    printf("Weight Adjusting time: %fms\n", cutGetTimerValue(hTimer));
  #endif


}



void CUDASaveProjection(CUDALISSOM *a, int w, int h, int i, FILE *fp) {
  CUDAPROJECTION *proj=a->projections[i];

  writeint(proj->type, fp);
  writeint(proj->afferentnum, fp);

  writefloat(proj->rf, fp);
  writefloat(proj->gamma, fp);
  writefloat(proj->alpha, fp);



  CUDAWEIGHT *weights=(CUDAWEIGHT *)malloc(proj->weightssize *sizeof(CUDAWEIGHT));
  cudaMemcpy( weights, proj->weights, proj->weightssize*sizeof(CUDAWEIGHT), cudaMemcpyDeviceToHost );  //TODO: OR LOAD GRADUALLY, WITHIN THE FOLLOWING LOOP?


  long long size=0;

  for(int p=0; p<w*h; p++) {
    int count = 0;

    for(int k=0; k<proj->numreceptors_host[p]; k++) {
      if(weights[proj->startindex_host[p] + k].x < 649999.0f  && weights[proj->startindex_host[p] + k].y>=0.00005 ) {
        count++;
      }
    }

    size += count;
  }


  writelong(size, fp); //proj->weightssize!!!



  size=0;

  for(int p=0; p<w*h; p++) {
    int count = 0;

    int start = proj->startindex_host[p];


    for(int k=0; k<proj->numreceptors_host[p]; k++) {
      if(weights[start + k].x < 649999.0f   && weights[proj->startindex_host[p] + k].y>=0.00005) {
        count++;
      }
    }

    writelong(size, fp); //startindex
    size+=count;


    writeint(count, fp); //numreceptors


    for(int k=0; k<proj->numreceptors_host[p]; k++) {
      if(weights[start + k].x < 649999.0f   && weights[proj->startindex_host[p] + k].y>=0.00005) {
        writefloat(weights[start + k].x, fp);
        writefloat(weights[start + k].y, fp);
      }
    }


  }


  free(weights);
}



void CUDASave(CUDALISSOM *a, int w, int h, int numinputs, FILE *fp) {
  int numproj = numinputs+2;

  for(int i=0; i<numproj; i++) {
    CUDASaveProjection(a, w, h, i, fp);
  }

}



void CUDALoadProjection(CUDALISSOM *a, int w, int h, int i, FILE *fp) {
  a->projections[i] = (CUDAPROJECTION *)malloc(sizeof(CUDAPROJECTION));

  CUDAPROJECTION *proj=a->projections[i];

  readint(proj->type, fp);
  readint(proj->afferentnum, fp);

  readfloat(proj->rf, fp);
  readfloat(proj->gamma, fp);
  readfloat(proj->alpha, fp);

  readlong(proj->weightssize, fp);


  proj->numreceptors_host=(int *)malloc(w*h *sizeof(int));
  proj->startindex_host=(long long *)malloc(w*h*sizeof(long long));

  cudaMalloc( (void **)&(proj->numreceptors), w*h *sizeof(int) );
  cudaMalloc( (void **)&(proj->startindex), w*h *sizeof(long long) );

  cudaMalloc( (void **)&(proj->weights), proj->weightssize*sizeof(CUDAWEIGHT) );



  CUDAWEIGHT *weights=(CUDAWEIGHT *)malloc(proj->weightssize *sizeof(CUDAWEIGHT));
  cudaMemcpy( weights, proj->weights, proj->weightssize*sizeof(CUDAWEIGHT), cudaMemcpyDeviceToHost );  //TODO: OR LOAD GRADUALLY, WITHIN THE FOLLOWING LOOP?

  for(int p=0; p<w*h; p++) {
    readlong(proj->startindex_host[p], fp); //startindex

    int start = proj->startindex_host[p];

    readint(proj->numreceptors_host[p], fp); //numreceptors


    for(int k=0; k<proj->numreceptors_host[p]; k++) {
      readfloat(weights[start + k].x, fp);
      readfloat(weights[start + k].y, fp);
    }


  }

  cudaMemcpy( proj->weights, weights, proj->weightssize*sizeof(CUDAWEIGHT), cudaMemcpyHostToDevice );  //TODO: OR LOAD GRADUALLY, WITHIN THE FOLLOWING LOOP?

  free(weights);


  cudaMemcpy( proj->numreceptors, proj->numreceptors_host, w*h*sizeof(int), cudaMemcpyHostToDevice );
  cudaMemcpy( proj->startindex, proj->startindex_host, w*h*sizeof(long long), cudaMemcpyHostToDevice );


}



void CUDALoad(CUDALISSOM *a, int w, int h, int numinputs, FILE *fp) {
  for(int i=0; i<numinputs+2; i++) {
    CUDALoadProjection(a, w, h, i, fp);
  }

}



void CUDASetAfferent(CUDALISSOM *dst, CUDALISSOM *src, int afferentnum) {
  dst->inputs_host[afferentnum] = src->neurons;
}
























}









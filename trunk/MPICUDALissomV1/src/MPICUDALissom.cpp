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

#include <cuda_runtime.h>

#include <boost/bind.hpp>
using namespace boost;

#include <iostream>
using namespace std;

#include "GPUWorker.h"


#include "MPICUDALissom.h"


#include "../../liblissom_cuda/CUDA/cudaLissom.h"
#include "../../liblissom_cuda/C++/lissom.h"



#define ROUND(x) ( x-floor(x)>0.5 ? (int)x+1 : (int)x )
#ifndef MAX
#define MAX(a, b) ( a>=b ? a : b )
#endif

#define writeint(a__, fp__) fwrite(&a__, sizeof(int), 1, fp__);
#define writefloat(a__, fp__) fwrite(&a__, sizeof(float), 1, fp__)

#define readint(a__, fp__) fread(&a__, sizeof(int), 1, fp__);
#define readfloat(a__, fp__) fread(&a__, sizeof(float), 1, fp__)




typedef struct __otherparams__ {
  int scaleAOD;
  float rE_, rI_, rEf_, alphaA_, alphaE_, alphaI_, gammaE_, gammaI_;
  int settletime_;
  float lowerthr_, upperthr_;
};


cudaError_t newLISSOM_(LISSOM **l, int w, int h, float rf, MPIRetina *retina, int numinputs, int weightsup, int weightsdown, __otherparams__ o) {
  *l=new LISSOM(w, h, rf, retina, numinputs, o.scaleAOD, weightsup, weightsdown, o.rE_, o.rI_, o.rEf_, o.alphaA_, o.alphaE_, o.alphaI_, o.gammaE_, o.gammaI_, o.settletime_, o.lowerthr_, o.upperthr_);

  return cudaSuccess;
}



cudaError_t deleteLISSOM_(LISSOM **l) {
  delete *l;

  return cudaSuccess;
}



cudaError_t setAfferent1_(CUDALISSOM **cuda, CUDALISSOM **Affcuda, int afferentnum) {
  CUDASetAfferent(*cuda, *Affcuda, afferentnum);

  return cudaSuccess;
}



cudaError_t setAfferent2_(CUDALISSOM **cuda, float **input, int afferentnum) {
  CUDALISSOM *cu = *cuda;
  cu->inputs_host[afferentnum] = *input;

  return cudaSuccess;
}



cudaError_t normalizeweights_(LISSOM **l) {
  LISSOM *ll=*l;
  ll->normalizeweights();

  return cudaSuccess;
}



cudaError_t firststep_(LISSOM **l) {
  LISSOM *ll=*l;
  ll->FirstStep();

  return cudaSuccess;
}



cudaError_t getoutput_(LISSOM **l, unsigned char *im, int widthstep, int w, int h) {
  LISSOM *ll=*l;
  CUDAGetOutput((CUDALISSOM *)(ll->cuda), im, widthstep, w, h);

  return cudaSuccess;
}



cudaError_t step__(LISSOM **l, int iters) {
  LISSOM *ll=*l;
  ll->Step(iters);

  return cudaSuccess;
}



cudaError_t adjustweights_(LISSOM **l) {
  LISSOM *ll=*l;
  ll->AdjustWeights();

  return cudaSuccess;
}



cudaError_t getweights_(unsigned char *im, int num, int x, int y, int widthstep, LISSOM **l) {
  LISSOM *ll=*l;
  ll->getweight(im, num, x, y, widthstep);

  return cudaSuccess;
}










MPICUDALissom::MPICUDALissom(int w_, int h_, float rf_, GPUWorker *context0, MPIRetina *afferent, int numinputs_, int scaleAreaOrDensity, int weightsup, int weightsdow, float rE_, float rI_, float rEf_, float alphaA_, float alphaE_, float alphaI_, float gammaE_, float gammaI_, int settletime_, float lowerthr_, float upperthr_) {
  //weightsup/weightsdown for first-last mpicudalissom object in a mpi hierarchy OR just for compatibility

  w=w_;
  h=h_;

  numinputs = numinputs_;

  g[0] = context0;

  g[0]->call(bind( cudaGetDeviceCount, &numGPUs ));
  if(numGPUs>=MAX_CPU_THREADS) numGPUs = MAX_CPU_THREADS;


  hPerGPU = h/numGPUs;


  retinas = (MPIRetina **)malloc(numinputs*sizeof(MPIRetina *));

  buf = (float *)malloc(w*h*sizeof(float));



  __otherparams__ o;
  o.scaleAOD=scaleAreaOrDensity;
  o.rE_=rE_;
  o.rI_=rI_;
  o.rEf_=rEf_;
  o.alphaA_=alphaA_;
  o.alphaE_=alphaE_;
  o.alphaI_=alphaI_;
  o.gammaE_=gammaE_;
  o.gammaI_=gammaI_;
  o.settletime_=settletime_;
  o.lowerthr_=lowerthr_;
  o.upperthr_=upperthr_;


  if(numGPUs==1) {
    g[0]->call(bind( newLISSOM_, &(layers[0]), w, hPerGPU, rf_, afferent, numinputs, 0, 0, o ));
  } else if(numGPUs==2) {
    g[1] = new GPUWorker(1);

    g[0]->call(bind( newLISSOM_, &(layers[0]), w, hPerGPU, rf_, afferent, numinputs, 0, 1, o ));

    g[1]->call(bind( newLISSOM_, &(layers[1]), w, hPerGPU, rf_, afferent, numinputs, 1, 0, o ));

  } else {
    g[0]->call(bind( newLISSOM_, &(layers[0]), w, hPerGPU, rf_, afferent, numinputs, 0, 1, o ));

    for(int i=1; i<numGPUs-1; i++) {
      g[i] = new GPUWorker(i);

      g[i]->call(bind( newLISSOM_, &(layers[i]), w, hPerGPU, rf_, afferent, numinputs, 1, 1, o ));
    }

    g[numGPUs-1] = new GPUWorker(numGPUs-1);
    g[numGPUs-1]->call(bind( newLISSOM_, &(layers[numGPUs-1]), w, hPerGPU, rf_, afferent, numinputs, 1, 0, o ));
  }


  //Free maps and allocate new ones. Shift them with some offset (record it).
  //On deleting, reshift them in place.
  for(int i=0; i<numGPUs; i++) {
    CUDALISSOM *c=(CUDALISSOM *)layers[i]->cuda;

    g[i]->call(bind( cudaFree, c->neurons ));
    g[i]->call(bind( cudaFree, c->temp ));
    g[i]->call(bind( cudaFree, c->firstactivation ));

    g[i]->call(bind( cudaMalloc, (void **)&(c->neurons), w*h*sizeof(float) ));
    g[i]->call(bind( cudaMalloc, (void **)&(c->temp), w*h*sizeof(float) ));
    g[i]->call(bind( cudaMalloc, (void **)&(c->firstactivation), w*h*sizeof(float) ));

    offsets[i] = i*h/numGPUs;
    offsets[i] *= w;

    c->neurons += offsets[i];
    c->temp += offsets[i];
    c->firstactivation += offsets[i];

  }


}



MPICUDALissom::~MPICUDALissom() {
  free(retinas);
  free(buf);

  for(int i=0; i<numGPUs; i++) {
    CUDALISSOM *c=(CUDALISSOM *)layers[i]->cuda;

    c->neurons -= offsets[i];
    c->temp -= offsets[i];
    c->firstactivation -= offsets[i];

    g[i]->call(bind( deleteLISSOM_, &layers[i] ));
  }

}



void MPICUDALissom::ConnectAfferent(MPIRetina *afferent, int afferentnum) {
  if(afferentnum>=0 && afferentnum<numinputs) {
    if(afferent->type==LAYER_RETINA) {
      retinas[afferentnum] = afferent;

      for(int i=1; i<numGPUs; i++) {
        afferent->AllocateOnGPU(g[i], i);
      }


      for(int i=0; i<numGPUs; i++) {
        layers[i]->inputw=afferent->w;
        layers[i]->inputh=afferent->h;

        layers[i]->inputWGPU=afferent->w;

        float w0_=layers[i]->inputw-2*layers[i]->rf;
        float h0_=layers[i]->inputh-2*layers[i]->rf;
        layers[i]->ratioW=(float)w0_/(float)w;
        layers[i]->ratioH=(float)h0_/(float)h;


        CUDALISSOM *c=(CUDALISSOM *)layers[i]->cuda;
        CUDALISSOM *aff=(CUDALISSOM *)afferent->cuda;
        if(i==0) {
          g[i]->call(bind( setAfferent1_, &c, &aff, afferentnum ));
        } else {
          g[i]->call(bind( setAfferent2_, &c, &(afferent->others[i]), afferentnum ));
        }

      }

    }


  }
}



void MPICUDALissom::normalizeweights() {
  for(int i=0; i<numGPUs; i++) {
    g[i]->call(bind( normalizeweights_, &layers[i] ));
  }

}



void MPICUDALissom::FirstStep() {
  for(int i=1; i<numGPUs; i++) {
    //TODO: copy for every input retina!!!
    retinas[0]->CopyToGPU(g[i], i);
  }

  for(int i=0; i<numGPUs; i++) {
    g[i]->call(bind( firststep_, &layers[i] ));

  }


}



void MPICUDALissom::getoutput(unsigned char *im, int widthstep) {
  if(widthstep==0) widthstep=w*sizeof(float);

  for(int i=0; i<numGPUs; i++) {
    //layers[i]->cuda (neurons)  --->   im+offset*widthstep

    g[i]->call(bind( getoutput_, &layers[i], im+offsets[i]*widthstep, widthstep, w, h));
  }

}



void MPICUDALissom::Step(int iters) {
  for(int i=0; i<numGPUs; i++) {
    g[i]->call(bind( step__, &layers[i], iters ));
  }

}



void MPICUDALissom::AdjustWeights() {
  for(int i=0; i<numGPUs; i++) {
    g[i]->call(bind( adjustweights_, &layers[i] ));
  }

}



void MPICUDALissom::getweight(unsigned char *im, int num, int x, int y, int widthstep) {
  int gpu=(int)(y/hPerGPU);

  g[gpu]->call(bind( getweights_, im, num, x, y, widthstep, &layers[gpu] ));

}




MPICUDALissom::MPICUDALissom(char *file) {


}



void MPICUDALissom::save(char *file) {
  FILE *fp=fopen(file, "wb");

  writeint(layers[0]->w, fp);
  writeint(layers[0]->h, fp);

  writefloat(layers[0]->rE, fp);
  writefloat(layers[0]->rI, fp);
  writefloat(layers[0]->rEf, fp);

  writefloat(layers[0]->gammaE, fp);
  writefloat(layers[0]->gammaI, fp);

  writefloat(layers[0]->alphaA, fp);
  writefloat(layers[0]->alphaE, fp);
  writefloat(layers[0]->alphaI, fp);

  writeint(layers[0]->inputw, fp);
  writeint(layers[0]->inputh, fp);

  writefloat(layers[0]->ratioW, fp);
  writefloat(layers[0]->ratioH, fp);

  writeint(layers[0]->numinputs, fp);
  writeint(layers[0]->inputWGPU, fp);

  writeint(layers[0]->settletime, fp);

  writeint(layers[0]->patternsperiteration, fp);

  writefloat(layers[0]->lowerthr, fp);
  writefloat(layers[0]->upperthr, fp);


//  CUDASave((CUDALISSOM *)cuda, w, h, numinputs, fp);

  int numproj = numinputs+2;

  for(int i=0; i<numproj; i++) {
//    CUDASaveProjection(a, w, h, i, fp);
    //Save taking weights from every LISSOM->cuda object
    CUDALISSOM *cu=(CUDALISSOM *)layers[0]->cuda;

    CUDAPROJECTION *proj=cu->projections[i];


    writeint(proj->type, fp);
    writeint(proj->afferentnum, fp);

    writefloat(proj->rf, fp);
    writefloat(proj->gamma, fp);
    writefloat(proj->alpha, fp);


    CUDAWEIGHT *weights[MAX_CPU_THREADS];


    unsigned int size=0;

    for(int j=0; j<numGPUs; j++) {
      cu=(CUDALISSOM *)layers[j]->cuda;
      proj=cu->projections[i];

      weights[j]=(CUDAWEIGHT *)malloc(proj->weightssize *sizeof(CUDAWEIGHT));

      g[j]->call(bind( cudaMemcpy, weights[j], proj->weights, proj->weightssize*sizeof(CUDAWEIGHT), cudaMemcpyDeviceToHost ));


      for(int p=0; p<layers[0]->w * layers[0]->h; p++) {
        int count=0;

        for(int kk=0; kk<proj->numreceptors_host[p]; kk++) {
          if(weights[j][proj->startindex_host[p] + kk].x < 64999.0f && weights[j][proj->startindex_host[p] + kk].y>=0.00005 ) {
            count++;
          }
        }

        size += count;
      }

    }


    writeint(size, fp); //proj->weightssize!!!

    size=0;



    for(int j=0; j<numGPUs; j++) {
      cu=(CUDALISSOM *)layers[j]->cuda;
      proj=cu->projections[i];


      for(int p=0; p<layers[0]->w * layers[0]->h; p++) {
        int count=0;

        int start=proj->startindex_host[p];

        for(int kk=0; kk<proj->numreceptors_host[p]; kk++) {
          if(weights[j][start + kk].x < 64999.0f   && weights[j][proj->startindex_host[p] + kk].y>=0.00005) {
            count++;
          }
        }

        writeint(size, fp); //startindex
        size+=count;

        writeint(count, fp); //numreceptors

        for(int kk=0; kk<proj->numreceptors_host[p]; kk++) {
          if(weights[j][start + kk].x < 64999.0f   && weights[j][proj->startindex_host[p] + kk].y>=0.00005) {
            writefloat(weights[j][start + kk].x, fp);
            writefloat(weights[j][start + kk].y, fp);
          }
        }

      }

    }



    for(int j=0; j<numGPUs; j++) {
      free(weights[j]);
    }
  }


  fclose(fp);
}




















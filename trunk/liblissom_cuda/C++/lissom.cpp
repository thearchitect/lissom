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
#include <time.h>
#include <math.h>

#include <cutil.h>
#include <cuda_runtime.h>


#include "lissom.h"

#include "../CUDA/cudaLissom.h"


#define ROUND(x) ( x-floor(x)>0.5 ? (int)x+1 : (int)x )


#define writeint(a__, fp__) fwrite(&a__, sizeof(int), 1, fp__);
#define writefloat(a__, fp__) fwrite(&a__, sizeof(float), 1, fp__)

#define readint(a__, fp__) fread(&a__, sizeof(int), 1, fp__);
#define readfloat(a__, fp__) fread(&a__, sizeof(float), 1, fp__)



LISSOM::LISSOM(int w_, int h_, float rf_, Layer *afferent, int numinputs_, int scaleAreaOrDensity, int weightsup, int weightsdown, int offsety, float offsetyAff, int realh, float rE_, float rI_, float rEf_, float alphaA_, float alphaE_, float alphaI_, float gammaE_, float gammaI_, int settletime_, float lowerthr_, float upperthr_) {
  w=w_;
  h=h_;
  rf=rf_;
  inputw=afferent->w;
  inputh=afferent->h;
  numinputs=numinputs_;

  type=LAYER_LISSOM;

  settletime=settletime_;
  lowerthr=lowerthr_;
  upperthr=upperthr_;

  patternsperiteration=1;



  if(rE_==0.0) {
//    rE_=(float)w/10.0;
    rE_=5.0;
  }

  if(rI_==0.0) {
//    rI_=(float)w/4.0 - 1.0;// *2.08);
    rI_=11.5;
  }

  if(scaleAreaOrDensity==1) {
    if(rE_==5.0) rE_=scaledExcRadius(w);
    if(rI_==11.5) rI_=scaledInhibRadius(w);
    if(rEf_==1.13) rEf_=scaledFinalExcRadius(w);
  } else if(scaleAreaOrDensity==0) {
//    if(patternsperiteration==1) patternsperiteration=scaledPatternsPerIteration(inputw, rf);
  }


  rE=rE_;
  rI=rI_;
//  rEf=MAX(1.5, (float)w/44.0);
  rEf=MAX(1.5, rEf_);

  alphaA=alphaA_;
  alphaE=alphaE_;
  alphaI=alphaI_;

  gammaE=gammaE_;
  gammaI=gammaI_;


//TODO: check on max Re, Ri!!! (over a certain number it's exaggereted, at least for internally generated parameters (that is, the values not passed to the constucted but calculated above))



  float w0_=inputw-2*rf;
  float h0_=inputh-2*rf;
  ratioW=(float)w0_/(float)w;
  ratioH=(float)h0_/(float)h;


  if(realh==0) realh=h;



  cuda=NewCUDALISSOM(w, h, inputw, inputh, &inputWGPU, numinputs, rf, rE, rI, alphaA, alphaE, alphaI, ratioW, ratioH, rf, gammaE, -gammaI, weightsup, weightsdown, 0, offsety, offsetyAff, realh);


}



LISSOM::LISSOM(char *file) {
  FILE *fp=fopen(file, "rb");

  type=LAYER_LISSOM;

  readint(w, fp);
  readint(h, fp);

  readfloat(rE, fp);
  readfloat(rI, fp);
  readfloat(rEf, fp);

  readfloat(gammaE, fp);
  readfloat(gammaI, fp);

  readfloat(alphaA, fp);
  readfloat(alphaE, fp);
  readfloat(alphaI, fp);

  readint(inputw, fp);
  readint(inputh, fp);

  readfloat(ratioW, fp);
  readfloat(ratioH, fp);

  readint(numinputs, fp);
  readint(inputWGPU, fp);

  readint(settletime, fp);

  readint(patternsperiteration, fp);

  readfloat(lowerthr, fp);
  readfloat(upperthr, fp);


  cuda=NewCUDALISSOM(w, h, inputw, inputh, &inputWGPU, numinputs, rf, rE, rI, alphaA, alphaE, alphaI, ratioW, ratioH, rf, gammaE, -gammaI, 1);



  CUDALoad((CUDALISSOM *)cuda, w, h, numinputs, fp);


  fclose(fp);
}



void LISSOM::load(char *file) {
  DeleteCUDALISSOM((CUDALISSOM *)cuda, numinputs, w, h);


  FILE *fp=fopen(file, "rb");

  type=LAYER_LISSOM;

  readint(w, fp);
  readint(h, fp);

  readfloat(rE, fp);
  readfloat(rI, fp);
  readfloat(rEf, fp);

  readfloat(gammaE, fp);
  readfloat(gammaI, fp);

  readfloat(alphaA, fp);
  readfloat(alphaE, fp);
  readfloat(alphaI, fp);

  readint(inputw, fp);
  readint(inputh, fp);

  readfloat(ratioW, fp);
  readfloat(ratioH, fp);

  readint(numinputs, fp);
  readint(inputWGPU, fp);

  readint(settletime, fp);

  readint(patternsperiteration, fp);

  readfloat(lowerthr, fp);
  readfloat(upperthr, fp);


  cuda=NewCUDALISSOM(w, h, inputw, inputh, &inputWGPU, numinputs, rf, rE, rI, alphaA, alphaE, alphaI, ratioW, ratioH, rf, gammaE, -gammaI, 1);



  CUDALoad((CUDALISSOM *)cuda, w, h, numinputs, fp);


  fclose(fp);
}



void LISSOM::save(char *file) {
  FILE *fp=fopen(file, "wb");

  writeint(w, fp);
  writeint(h, fp);

  writefloat(rE, fp);
  writefloat(rI, fp);
  writefloat(rEf, fp);

  writefloat(gammaE, fp);
  writefloat(gammaI, fp);

  writefloat(alphaA, fp);
  writefloat(alphaE, fp);
  writefloat(alphaI, fp);

  writeint(inputw, fp);
  writeint(inputh, fp);

  writefloat(ratioW, fp);
  writefloat(ratioH, fp);

  writeint(numinputs, fp);
  writeint(inputWGPU, fp);

  writeint(settletime, fp);

  writeint(patternsperiteration, fp);

  writefloat(lowerthr, fp);
  writefloat(upperthr, fp);


  CUDASave((CUDALISSOM *)cuda, w, h, numinputs, fp);


  fclose(fp);
}



LISSOM::~LISSOM() {

  DeleteCUDALISSOM((CUDALISSOM *)cuda, numinputs, w, h);
}



void LISSOM::getoutput(unsigned char *im, int widthstep) {
  if(widthstep==0) widthstep=w*sizeof(float);

  CUDAGetOutput((CUDALISSOM *)cuda, im, widthstep, w, h);
}



void LISSOM::ConnectAfferent(Layer *afferent, int afferentnum) {
  if(afferentnum>=0 && afferentnum<numinputs) {
    if(afferent->type==LAYER_RETINA || afferent->type==LAYER_LISSOM) {
      inputw=afferent->w;
      inputh=afferent->h;

      inputWGPU=afferent->w;

//      afferent->patternsperiteration=patternsperiteration;

      float w0_=inputw-2*rf;
      float h0_=inputh-2*rf;
      ratioW=(float)w0_/(float)w;
      ratioH=(float)h0_/(float)h;

      CUDASetAfferent((CUDALISSOM *)cuda, (CUDALISSOM *)afferent->cuda, afferentnum);

    } else if(afferent->type==LAYER_LGN) {
      if(afferentnum>=numinputs-1) return;

      inputw=afferent->w;
      inputh=afferent->h;

      inputWGPU=afferent->w;

      afferent->patternsperiteration=patternsperiteration;

      float w0_=inputw-2*rf;
      float h0_=inputh-2*rf;
      ratioW=(float)w0_/(float)w;
      ratioH=(float)h0_/(float)h;

      CUDASetAfferentLGN((CUDALISSOM *)cuda, (CUDALISSOM *)afferent->cuda, afferentnum);

    }

  }
}



void LISSOM::FirstStep() {
  CUDAFirstStep((CUDALISSOM *)cuda, w, h, lowerthr, upperthr, numinputs, inputWGPU, inputh);


}



void LISSOM::Step(int iters) {
  if(iters==-1) iters=settletime;

  for(int o=0; o<iters; o++) {
    CUDAStep((CUDALISSOM *)cuda, w, h, lowerthr, upperthr);


  }

}



void LISSOM::setThresholds(float lower, float upper) {
  lowerthr=lower;
  upperthr=upper;
}



void LISSOM::setGammas(float E, float I) {
  gammaE=E;
  gammaI=I;

  CUDASetGammas((CUDALISSOM *)cuda, E, I);
}



void LISSOM::setAlphaA(float a) {
  alphaA=a;

  CUDASetAlphaA((CUDALISSOM *)cuda, a, numinputs);
}



void LISSOM::setAlphaE(float a) {
  alphaE=a;

  CUDASetAlphaE((CUDALISSOM *)cuda, a);
}



void LISSOM::setAlphaI(float a) {
  alphaI=a;

  CUDASetAlphaI((CUDALISSOM *)cuda, a);
}



void LISSOM::setRef(float r) {
  rEf=r;
}



void LISSOM::normalizeweights() {
  CUDANormalizeWeights((CUDALISSOM *)cuda, numinputs, w, h);


}



void LISSOM::setRe(float r, int offsety) {
  if(r>=rEf) {
    rE=r;

    CUDASetRe((CUDALISSOM *)cuda, r, w, h, offsety);


    this->normalizeweights();
  }

}



void LISSOM::AdjustWeights() {
  CUDAAdjustWeights((CUDALISSOM *)cuda, w, h, inputWGPU, inputh, numinputs);


}



void LISSOM::getweight(unsigned char *im, int num, int x, int y, int widthstep) {
  if(widthstep==0) widthstep=inputw*sizeof(float);

  CUDAGetWeight((CUDALISSOM *)cuda, im, widthstep, num, x, y, w, h, inputw, inputh);

}






float scaledExcRadius(int w) {
  return (float)w/10.0;
}



float scaledInhibRadius(int w) {
  return (float)w/4.0 - 1.0;
}



float scaledFinalExcRadius(int w) {
  return MAX(2.5, (float)w/44.0);
}



int scaledPatternsPerIteration(int w, float rf) {
  float tmp=((float)w - 2.0f*rf)/(36.0f - 2.0*rf);
  return (int)ROUND(tmp*tmp);
}
















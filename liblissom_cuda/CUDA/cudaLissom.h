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

#ifndef __CUDALISSOM__H__
#define __CUDALISSOM__H__


#include <vector_types.h>


#define THREADS 64
#define BLOCKS 1024
//BLOCKS MAY BE SOMETHING LIKE 128/256 ON TESLA



extern "C" {
//#define DEBUG


/*
#ifdef __CUDACC__
typedef struct __align__(8)
#else
typedef struct
#endif
{
  unsigned short x, y;
  float val;
} CUDAWEIGHT;
*/

typedef float2 CUDAWEIGHT;



#define PROJECTION_AFFERENT 0
#define PROJECTION_EXCITATORY 1
#define PROJECTION_INHIBITORY 2

typedef struct {
  CUDAWEIGHT *weights; //weights[startindex + index].x/y/z

  long long *startindex_host;
  long long *startindex;
  int *numreceptors_host;
  int *numreceptors;

  long long weightssize;

  int type;
  int afferentnum;
  float rf;
  float gamma;
  float alpha;
} CUDAPROJECTION;



typedef struct {
  float **inputs_host;
//  float **inputs;
  float *neurons;
  float *temp;
  float *firstactivation;


  CUDAPROJECTION **projections;
} CUDALISSOM;




CUDALISSOM *NewCUDALISSOM(int w, int h, int inputw, int inputh, unsigned int *inputWGPU, int numinputs, float rA, float rE, float rI, float alphaA, float alphaE, float alphaI, float ratioW, float ratioH, float offset, float gammaE, float gammaI, int weightsup=0, int weightsdown=0, int donotinit=0, int offsety=0, float offsetyAff=0.0, int realh=0);
void DeleteCUDALISSOM(CUDALISSOM *a, int numinputs, int w, int h);

CUDAPROJECTION *NewCUDAPROJECTION(int w, int h, int type, float rf, float alpha, float ratioW, float ratioH, float offset, float gamma, int weightsup, int weightsdown, int afferentnum=0);
void DeleteCUDAPROJECTION(CUDAPROJECTION *a, int w, int h);

//void CUDASetInput(CUDALISSOM *a, unsigned char *im, int widthstep, int inputnum, unsigned int inputWGPU, int inputw, int inputh);
//void CUDAGetInput(CUDALISSOM *a, unsigned char *im, int widthstep, int inputnum, unsigned int inputWGPU, int inputw, int inputh);
void CUDAGetOutput(CUDALISSOM *a, unsigned char *im, int widthstep, int w, int h);
void CUDAGetWeight(CUDALISSOM *a, unsigned char *im, int widthstep, int num, int x, int y, int w, int h, int inputw, int inputh, int offsety=0, float ratio=1.0f);

void CUDARandomGaussian(CUDALISSOM *a, int centered, float a2, float b2, unsigned int inputWGPU, int inputw, int inputh, int number=1);

void CUDAFirstStep(CUDALISSOM *a, int w, int h, float lowerthr, float upperthr, int numinputs, unsigned int inputWGPU, int inputh);

void CUDAStep(CUDALISSOM *a, int w, int h, float lowerthr, float upperthr, int realh=0); //realh is for use with MPICUDALissom

void CUDASetGammas(CUDALISSOM *a, float E, float I);
void CUDASetAlphaA(CUDALISSOM *b, float a, int numinputs);
void CUDASetAlphaE(CUDALISSOM *b, float a);
void CUDASetAlphaI(CUDALISSOM *b, float a);

void CUDANormalizeWeights(CUDALISSOM *a, int numinputs, int w, int h);

void CUDASetRe(CUDALISSOM *a, float r, int w, int h, int offsety=0);

void CUDAAdjustWeights(CUDALISSOM *a, int w, int h, int inputWGPU, int inputh, int numinputs, int realh=0, int offsety=0);

void CUDASave(CUDALISSOM *a, int w, int h, int numinputs, FILE *fp);
void CUDALoad(CUDALISSOM *a, int w, int h, int numinputs, FILE *fp);



CUDALISSOM *CUDANewRetina(int w, int h);
void CUDADeleteRetina(CUDALISSOM *a);

void CUDARetinaSetInput(CUDALISSOM *a, unsigned char *im, int widthstep, int w, int h);
void CUDARetinaRandomGaussian(CUDALISSOM *a, int centered, float a2, float b2, int w, int h, int number, int x, int y, int angledeg, float thr);
void CUDARetinaOrientedBar(CUDALISSOM *a, int w, int h, float m, float q, float a2, float aa, float thr);


CUDALISSOM *CUDANewLGN(int w, int h);
void CUDADeleteLGN(CUDALISSOM *a);


void CUDASetAfferent(CUDALISSOM *dst, CUDALISSOM *src, int afferentnum);
void CUDAGetBuffer(float *buf, unsigned char *im, int widthstep, int w, int h);
void CUDALGNRun(CUDALISSOM *a, int w, int h, int inputw, int inputh, float ratioW, float ratioH, int rf);

void CUDASetAfferentLGN(CUDALISSOM *cuda, CUDALISSOM *aff, int startaffnum);





}


#endif //__CUDALISSOM__H__


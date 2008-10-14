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

#include "lissom.h"

#include "../CUDA/cudaLissom.h"



LGN::LGN(int w_, int h_) {
  w=w_;
  h=h_;

  rf=9;

  type=LAYER_LGN;

  cuda=(void *)CUDANewLGN(w, h);
}



LGN::~LGN() {
  CUDADeleteLGN((CUDALISSOM *)cuda);
}



void LGN::ConnectAfferent(Layer *afferent) {
  if(afferent->type==LAYER_RETINA) {
    patternsperiteration=afferent->patternsperiteration;
    inputw=afferent->w;
    inputh=afferent->h;

    float w0_=(float)(inputw-2*rf);
    float h0_=(float)(inputh-2*rf);
    ratioW=(float)w0_/(float)w;
    ratioH=(float)h0_/(float)h;

    CUDASetAfferent((CUDALISSOM *)cuda, (CUDALISSOM *)(afferent->cuda), 0);

  }

}



void LGN::run() {
  CUDALGNRun((CUDALISSOM *)cuda, w, h, inputw, inputh, ratioW, ratioH, rf);

}



void LGN::getoutput(unsigned char *im, int OnOff, int widthstep) {
  if(widthstep==0) widthstep=w*sizeof(float);

  CUDALISSOM *cu=(CUDALISSOM *)cuda;

  if(OnOff==0) CUDAGetBuffer(cu->neurons, im, widthstep, w, h);
  if(OnOff==1) CUDAGetBuffer(cu->temp, im, widthstep, w, h);

}












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



Retina::Retina(int w_, int h_, int patternsperiteration_) {
  w=w_;
  h=h_;
  patternsperiteration=patternsperiteration_;

  type=LAYER_RETINA;

  cuda=(void *)CUDANewRetina(w, h);
}



Retina::~Retina() {
  CUDADeleteRetina((CUDALISSOM *)cuda);
}



void Retina::setinput(unsigned char *im, int widthstep) {
  if(widthstep==0) widthstep=w*sizeof(float);

  CUDARetinaSetInput((CUDALISSOM *)cuda, im, widthstep, w, h);

}



void Retina::getoutput(unsigned char *im, int widthstep) {
  if(widthstep==0) widthstep=w*sizeof(float);

  CUDAGetOutput((CUDALISSOM *)cuda, im, widthstep, w, h);

}



void Retina::randomGaussian(int centered, int number, float a2, float b2, int x, int y, int angledeg, float thr) {
  if(number==0) number=patternsperiteration;
  CUDARetinaRandomGaussian((CUDALISSOM *)cuda, centered, a2, b2, w, h, number, x, y, angledeg, thr);

}




















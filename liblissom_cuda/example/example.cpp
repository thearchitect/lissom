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

#include "../C++/lissom.h"


#include <iostream>
using namespace std;


#include <cv.h>
#include <highgui.h>

#define N 50
#define RetinaN 36


int main() {
  IplImage *map=cvCreateImage(cvSize(N, N), IPL_DEPTH_32F, 1);
  IplImage *im=cvCreateImage(cvSize(RetinaN, RetinaN), IPL_DEPTH_32F, 1);


  Retina *retina = new Retina(RetinaN, RetinaN);

  LISSOM *layer = new LISSOM(N, N, 6.0, retina, 1, 1); //0
  float Rei=layer->rE, Ref=layer->rEf;

//layer->load("net.lissom");




  layer->ConnectAfferent(retina);




/*
for(int o=0; o<1000; o++) {
if(o%100==0) printf("%d\n", o);
  retina->randomGaussian();
  layer->FirstStep();
  layer->Step();

  layer->AdjustWeights();


if(o==200) {
  layer->setThresholds(layer->lowerthr+0.01, layer->upperthr+0.01);
  layer->setRe(MAX(Ref, 0.6*Rei));

} else if(o==500) {
  layer->setThresholds(layer->lowerthr+0.01, layer->upperthr+0.01);
  layer->setAlphaA(layer->alphaA*5.0/7.0);
  layer->setAlphaE(layer->alphaE*0.5);
  layer->setRe(MAX(Ref, 0.420*Rei));

} else if(o==1000) {
  layer->setThresholds(layer->lowerthr+0.03, layer->upperthr+0.01);
  layer->setRe(MAX(Ref, 0.336*Rei));

} else if(o==2000) {
  layer->setThresholds(layer->lowerthr+0.03, layer->upperthr+0.02);
  layer->setAlphaA(layer->alphaA*4.0/5.0);
  layer->setSettleTime(layer->settletime+1);
  layer->setRe(MAX(Ref, 0.269*Rei));

} else if(o==3000) {
  layer->setThresholds(layer->lowerthr+0.02, layer->upperthr+0.03);
  layer->setRe(MAX(Ref, 0.215*Rei));

} else if(o==4000) {
  layer->setThresholds(layer->lowerthr, layer->upperthr+0.03);
  layer->setAlphaA(layer->alphaA*3.0/4.0);
  layer->setRe(MAX(Ref, 0.129*Rei));

} else if(o==5000) {
  layer->setThresholds(layer->lowerthr+0.01, layer->upperthr+0.03);
  layer->setSettleTime(layer->settletime+1);
  layer->setRe(MAX(Ref, 0.077*Rei));

} else if(o==6500) {
  layer->setThresholds(layer->lowerthr+0.01, layer->upperthr+0.03);
  layer->setSettleTime(layer->settletime+1);
  layer->setRe(MAX(Ref, 0.046*Rei));

} else if(o==8000) {
  layer->setThresholds(layer->lowerthr+0.01, layer->upperthr+0.03);
  layer->setSettleTime(layer->settletime+1);
  layer->setRe(MAX(Ref, 0.028*Rei));

} else if(o==20000) {
  layer->setThresholds(layer->lowerthr+0.01, layer->upperthr+0.03);
  layer->setAlphaA(layer->alphaA*0.5);
  layer->setRe(MAX(Ref, 0.017*Rei));

}

}
/*
*/





  retina->randomGaussian(1);


  retina->getoutput((unsigned char *)im->imageData, im->widthStep);




//layer->AdjustWeights();


  layer->FirstStep();

  layer->Step();


//layer->getweight((unsigned char *)im->imageData, 1, 24, 24, im->widthStep);


  layer->getoutput((unsigned char *)map->imageData, map->widthStep);




//layer->save("net.lissom");





  delete retina;
  delete layer;


  cvNamedWindow("mainWin", CV_WINDOW_AUTOSIZE);
  cvShowImage("mainWin", map);

  cvNamedWindow("mainWin2", CV_WINDOW_AUTOSIZE);
  cvShowImage("mainWin2", im);

  cvWaitKey(0);


  cvReleaseImage(&im);
  cvReleaseImage(&map);

  return 0;
}













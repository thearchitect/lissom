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

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;


#include "../src/MPICUDALissom.h"
#include "../../liblissom_cuda/C++/lissom.h"


#include <cutil.h>

#include <cv.h>
#include <highgui.h>


int main() {
  IplImage *map=cvCreateImage(cvSize(50, 50), IPL_DEPTH_32F, 1);
  IplImage *im=cvCreateImage(cvSize(36, 36), IPL_DEPTH_32F, 1);


  GPUWorker *g0 = new GPUWorker(0);


  MPIRetina *retina = new MPIRetina(36, 36, g0);

  MPICUDALissom *layer = new MPICUDALissom(50, 50, 6.0, g0, retina, 1, 0);
  LISSOM *l0 = layer->layers[0];

  float Rei=l0->rE, Ref=l0->rEf;


  layer->ConnectAfferent(retina);



unsigned int hTimer;
cutCreateTimer(&hTimer);
cutStartTimer(hTimer);

/*
for(int o=0; o<20; o++) {
if(o%100==0) printf("%d\n", o);
  retina->randomGaussian();
  layer->FirstStep();
  layer->Step();

  layer->AdjustWeights();


if(o==200) {
  layer->setThresholds(l0->lowerthr+0.01, l0->upperthr+0.01);
  layer->setRe(MAX(Ref, 0.6*Rei));

} else if(o==500) {
  layer->setThresholds(l0->lowerthr+0.01, l0->upperthr+0.01);
  layer->setAlphaA(l0->alphaA*5.0/7.0);
  layer->setAlphaE(l0->alphaE*0.5);
  layer->setRe(MAX(Ref, 0.420*Rei));

} else if(o==1000) {
  layer->setThresholds(l0->lowerthr+0.03, l0->upperthr+0.01);
  layer->setRe(MAX(Ref, 0.336*Rei));

} else if(o==2000) {
  layer->setThresholds(l0->lowerthr+0.03, l0->upperthr+0.02);
  layer->setAlphaA(l0->alphaA*4.0/5.0);
  layer->setSettleTime(l0->settletime+1);
  layer->setRe(MAX(Ref, 0.269*Rei));

} else if(o==3000) {
  layer->setThresholds(l0->lowerthr+0.02, l0->upperthr+0.03);
  layer->setRe(MAX(Ref, 0.215*Rei));

} else if(o==4000) {
  layer->setThresholds(l0->lowerthr, l0->upperthr+0.03);
  layer->setAlphaA(l0->alphaA*3.0/4.0);
  layer->setRe(MAX(Ref, 0.129*Rei));

} else if(o==5000) {
  layer->setThresholds(l0->lowerthr+0.01, l0->upperthr+0.03);
  layer->setSettleTime(l0->settletime+1);
  layer->setRe(MAX(Ref, 0.077*Rei));

} else if(o==6500) {
  layer->setThresholds(l0->lowerthr+0.01, l0->upperthr+0.03);
  layer->setSettleTime(l0->settletime+1);
  layer->setRe(MAX(Ref, 0.046*Rei));

} else if(o==8000) {
  layer->setThresholds(l0->lowerthr+0.01, l0->upperthr+0.03);
  layer->setSettleTime(l0->settletime+1);
  layer->setRe(MAX(Ref, 0.028*Rei));

} else if(o==20000) {
  layer->setThresholds(l0->lowerthr+0.01, l0->upperthr+0.03);
  layer->setAlphaA(l0->alphaA*0.5);
  layer->setRe(MAX(Ref, 0.017*Rei));

}

}
/*
*/

cutStopTimer(hTimer);
printf("Training time: %fms\n", cutGetTimerValue(hTimer));





  retina->randomGaussian(1);

  retina->getoutput((unsigned char *)im->imageData, im->widthStep);



  layer->AdjustWeights();


  layer->FirstStep();


  layer->Step();



//layer->getweight((unsigned char *)im->imageData, 0, 24, 24, im->widthStep);


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









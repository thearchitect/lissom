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

#ifndef __LIBLISSOM__H__
#define __LIBLISSOM__H__


#ifndef MAX
#define MAX(a, b) ( a>=b ? a : b )
#endif


#define RETINATHR 0.2f


#define LAYER_LISSOM 0
#define LAYER_RETINA 1
#define LAYER_LGN 2

class Layer {
  public:
    int w, h;

    int type;

    int patternsperiteration; //used mainly by retina and lissom object. lgn should ignore it.

    void *cuda;
};


//TODO: save/load: after saving, you will have to load the map to work with an identical! retina (size). it would be quite hard to change weights' coords based on a new retina size.
class LISSOM : public Layer {
public:
  float rf;

  float rE, rI;
  float rEf;

  float gammaE, gammaI;

  float alphaA, alphaE, alphaI;

  int inputw, inputh;
  float ratioW, ratioH;

  int numinputs;

//  int patternsperiteration;

  unsigned int inputWGPU;

  int settletime;
  float lowerthr, upperthr;



  //scaleAreaOrDensity: 0 for area, 1 for density
  LISSOM(int w_, int h_, float rf_, Layer *afferent, int numinputs_=1, int scaleAreaOrDensity=0, int weightsup=0, int weightsdown=0, int offsety=0, float offsetyAff=0.0, float rE_=0.0, float rI_=0.0, float rEf_=1.13, float alphaA_=0.007, float alphaE_=0.0330078, float alphaI_=0.00466167, float gammaE_=0.9, float gammaI_=0.9, int settletime_=9, float lowerthr_=0.1, float upperthr_=0.65);

  ~LISSOM();

  LISSOM(char *file);
  void load(char *file );
  void save(char *file);

  void setThresholds(float lower, float upper);
  void setGammas(float E, float I);
  void setAlphaA(float a);
  void setAlphaE(float a);
  void setAlphaI(float a);
  void setRef(float r);
  void setRe(float r, int offsety=0);
  void setSettleTime(int t) { settletime=t; }

  void getweight(unsigned char *im, int num, int x, int y, int widthstep=0);

  void ConnectAfferent(Layer *afferent, int afferentnum=0);  //Be sure to check afferentnum!

  void getoutput(unsigned char *im, int widthstep=0);

  void normalizeweights();
  void FirstStep();
  void Step(int iters=-1);
  void AdjustWeights();


};



float scaledExcRadius(int w);
float scaledInhibRadius(int w);
float scaledFinalExcRadius(int w);
int scaledPatternsPerIteration(int w, float rf);
//TODO: learning weights (eg: alphas) scale (sometimes), too


class Retina : public Layer {
public:
//  int patternsperiteration;

  Retina(int w_, int h_, int patternsperiteration_=1);
  ~Retina();

  void setPatternsPerIteration(int i) { if(i>=1) patternsperiteration=i; } //Don't forget to set it!

  void setinput(unsigned char *im, int widthstep=0);
  void randomGaussian(int centered=0, int number=0, float a2=56.25, float b2=2.25, int x=-1, int y=-1, int angledeg=-1, float thr=RETINATHR);//0.369f);
  void OrientedBar(float m, float q, float a2=1.0, float thr=RETINATHR); //m==100 for vertical line
  //thrs = 1/e^2?  == 0.135
  void getoutput(unsigned char *im, int widthstep=0);

};



//LGN IS 2 INPUTS!!!
class LGN : public Layer {
public:
  int inputw, inputh;
  int rf;

  float ratioW, ratioH;


  LGN(int w_, int h_);
  ~LGN();

  void ConnectAfferent(Layer *afferent); //also get a copy of patternsperiteration
  void run();
  void getoutput(unsigned char *im, int OnOff=0, int widthstep=0); //0=on, 1=off
};








#endif /* __LIBLISSOM__H__ */



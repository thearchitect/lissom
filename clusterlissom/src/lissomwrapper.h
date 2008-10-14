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

#ifndef __LISSOMWRAPPER__H__
#define __LISSOMWRAPPER__H__


#include <iostream>
#include <string>
#include <vector>
using namespace std;


#include "parser.h"
#include "config.h"
#include "../lissom.h"



void parse(int *val, string name, vector<param> v);
void parse(float *val, string name, vector<param> v);



class afferentlayer {
public:
  Retina *layer;
  int id; //process owning the real object


  ~afferentlayer();

};




class LissomWrapper {
public:
  Retina *retina;
  LISSOM *layer;
  int isretina;


  unsigned char *obuf; //1st is for output
  int w, h;


  int action; // ACTIONTRAIN / ACTIONRUN

  //ids of maps to send data to/receive data from
  vector<int> in;
  vector<int> out;

  string mapname;
  int mid;

  int starttraining;
  int iterationstotrain;


  float initrexc;
  float finalrexc;


  vector<int> steps;
  vector<param> train; //param x;  x.step is the time


  vector<afferentlayer*> inp;
//Layer objects representing `Layer` afferents, to be filled up with data "manually" (within classes to do the job on their own?)





  void open(Parser p, int id);
  ~LissomWrapper();

  void getinput(string s);
  void setoutput(string s);



  //void load(char *f);
  //void save(char *f);

  void randomGaussian(int center=0);

  void FirstStep(); // -getinputmap- -process- -write output map to everybody else needing it-
  void Step();
  void AdjustWeights();






};






#endif


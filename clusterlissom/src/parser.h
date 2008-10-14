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

#ifndef __PARSER__H__
#define __PARSER__H__


#include <iostream>
#include <string>
#include <vector>

using namespace std;


#include "config.h"


#define ACTIONTRAIN 0
#define ACTIONRUN 1
#define ACTIONLOADRUN 2
#define ACTIONLOADTRAIN 3




class Parser {
public:
  //ACTION
  int action;


  //LINK
  vector<string> first;
  vector<string> second;


  //MAP/RETINA
  vector<string> mapid;
  vector<int> starttraining;

  vector<int> retinas; //ids in mapid of retinas

  vector<vector<param> > map;


  //TRAINING
  vector<int> steps; //steps time, eg: 200-300-500-800-1000-2000-...-20000
  vector<vector<param> > train;  //train[id]list of params] (using .step as time measure)


  void open(Config *c);
  void processstep(Config *c, int *type, param *p, int id);
};




#endif

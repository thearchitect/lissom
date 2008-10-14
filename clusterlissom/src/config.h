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

#ifndef __CONFIG__H__
#define __CONFIG__H__


#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define MAP 0
#define LINK 1
#define PARAMINT 2
#define PARAMFLOAT 3
#define RETINA 4
#define TRAINING 5
#define STEP 6
#define ACTION 7



typedef struct param {
  int i;
  float f;
  string s;
  string s2;

  int step;// N of step in which to execute (just for parsing `TRAINING`)
};


class Config {
private:
  ifstream conf;

public:
  ~Config();

  void open(char *file);
  void close();

  int get(param *p); //returns command type

};






#endif

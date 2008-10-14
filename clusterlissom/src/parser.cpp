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

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>

using namespace std;


#include "parser.h"
//#include "config.h"


void Parser::open(Config *c) {
  int reuse=0;
  int type=-1;
  int firsttimetraining=1;

  param p;
  while(1) {
    if(reuse==0) type=c->get(&p);
    else reuse=0;

    if(type==-1) break;

    if(type==ACTION) {
      if(p.s=="train") action=ACTIONTRAIN;
      else if(p.s=="run") action=ACTIONRUN;
      else if(p.s=="loadrun") action=ACTIONLOADRUN;
      else if(p.s=="loadtrain") action=ACTIONLOADTRAIN;
    } else if(type==LINK) {
      first.push_back(p.s);
      second.push_back(p.s2);
    } else if(type==MAP || type==RETINA) {
      mapid.push_back(p.s);

      if(type==RETINA) retinas.push_back(mapid.size()-1);


      //get id
      int id=0;
      for(id=0; id<mapid.size(); id++) {
        if(mapid[id]==p.s) break;
      }

      while(id>=map.size()) {
        vector<param> tmp;
        map.push_back(tmp);
      }

      type=PARAMINT;
      while(type==PARAMINT || type==PARAMFLOAT) {
        type=c->get(&p);

        if(type==PARAMINT) {
          p.f=-1000000.0;
        } else if(type==PARAMFLOAT) {
          p.i=-1000000;
        }
        if(type==PARAMINT || type==PARAMFLOAT) map[id].push_back(p);
      }
      reuse=1;


    } else if(type==TRAINING) {
      if(firsttimetraining==1) {
        for(int i=0; i<mapid.size(); i++) {
          vector<param> tmp;
          train.push_back(tmp);
          starttraining.push_back(0);
        }
      }


      //get id
      int id=0;


      for(id=0; id<mapid.size(); id++) {
        if(mapid[id]==p.s) break;
      }
      starttraining[id]=p.i;


      while(type!=STEP) {
        type=c->get(&p);
      }


      while(type==STEP) {
        processstep(c, &type, &p, id);
      }


    }



  }


  vector<int> numafferent;
  for(int i=0; i<mapid.size(); i++) numafferent.push_back(0);
  //number of inputs for each cortical map
  for(int i=0; i<second.size(); i++) {
    int id;
    for(id=0; id<mapid.size(); id++) {
      if(mapid[id]==second[i]) break;
    }

    numafferent[id]+=1;
  }
  for(int i=0; i<mapid.size(); i++) {
    param p;
    p.s="numafferent";
    p.i=numafferent[i];
    map[i].push_back(p);
  }


}




void Parser::processstep(Config *c, int *type, param *p, int id) {
  int n=p->i;
  steps.push_back(n);

  *type=PARAMINT;


  while(*type==PARAMINT || *type==PARAMFLOAT) {
    *type=c->get(p);


    if(*type==PARAMINT || *type==PARAMFLOAT) {
      p->step=n;
      train[id].push_back(*p);
    }

  }

}






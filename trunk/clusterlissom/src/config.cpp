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


#include "config.h"


Config::~Config() {
  close();
}



void Config::open(char *file) {
  conf.open(file);

  if(!conf) { cout<<"Error opening file"<<endl<<endl; exit(1); }
}



void Config::close() {
  if(conf) conf.close();
}



int Config::get(param *p) {
  string l="";

  p->i=0;
  p->f=0;
  p->s2="";


  while(!conf.eof() && l=="") {
    getline(conf, l);

    if(l!="") {
      size_t startpos = l.find_first_not_of(" \t");
      size_t endpos = l.find_last_not_of(" \t");

      if(( string::npos == startpos ) || ( string::npos == endpos)) l = "";
      else l = l.substr(startpos, endpos-startpos+1);
    }

    if(l.size()>0) if(l[0]=='#') l="";
  }
  if(conf.eof()) return -1;


  //3 types of data, `MAP`, `LINK` or parameters (lowercase)
  int type;
  if(l[0]=='M') {
    type=MAP;

    string name=l.substr(4, l.size()-3);

    (*p).s=name;

  } else if(l[0]=='R') {
    if(l[0]=='R') type=RETINA;

    string name=l.substr(7, l.size()-6);

    (*p).s=name;
  } else if(l[0]=='L') {
    type=LINK;

    string parm=l.substr(5, l.size()-4);

    string a, b;
    int div=parm.find(" ");
    a=parm.substr(0, div);
    b=parm.substr(div+1, parm.size()-div+1);

    (*p).s=a;
    (*p).s2=b;
  } else if(l[0]=='T') {
    type=TRAINING;

    string nm=l.substr(9, l.size()-8);
    int div=nm.find(" ");
    string name=nm.substr(0, div);
    string b=nm.substr(div+1, nm.size()-div+1);

    stringstream st(b);

    int n;
    st>>n;


    (*p).s=name;
    (*p).i=n;
  } else if(l[0]=='[') {
    type=STEP;

    string b=l.substr(6, l.size()-5);
    stringstream st(b);

    int n;
    st>>n;

    (*p).i=n;
  } else if(l[0]=='A') {
    type=ACTION;

    string b=l.substr(7, l.size()-7);

    (*p).s=b;
  }else {
    //a parameter,  select type

    string a, b;
    int div=l.find(" ");
    a=l.substr(0, div);
    b=l.substr(div+1, l.size()-div+1);

    stringstream st(b);

    if(b[0]=='+' || b[0]=='-' || b[0]=='*') {
      (*p).s2=b;
      type=PARAMINT; //with NO meaning (specified by `STEP` before)
    } else {
      if(b.find(".")!=string::npos) {
        type=PARAMFLOAT;
        st>>(*p).f;
      } else {
        type=PARAMINT;
        st>>(*p).i;
      }

    }

    (*p).s=a;

  }





  return type;
}







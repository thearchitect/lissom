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

#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR
#include "mpi.h"


using namespace std;


#include "parser.h"
#include "config.h"
#include "lissomwrapper.h"

#include "../lissom.h"



void parse(int *val, string name, vector<param> v) {
  *val=-1;
  for(int i=0; i<v.size(); i++) {
    if(v[i].s==name) {
      *val=v[i].i;
      return;
    }
  }

}


void parse(float *val, string name, vector<param> v) {
  *val=-1.0;
  for(int i=0; i<v.size(); i++) {
    if(v[i].s==name) {
      *val=v[i].f;
      return;
    }
  }

}




void LissomWrapper::open(Parser p, int id) {
  action=p.action; //ACTIONTRAIN / ACTIONRUN


  mapname=p.mapid[id];
  mid=id;


  out.clear();
  for(int i=0; i<p.first.size(); i++) {
    if(p.first[i]==mapname) {
      int tmp;
      for(tmp=0; tmp<p.mapid.size(); tmp++) {
        if(p.mapid[tmp]==p.second[i]) break;
      }
      out.push_back(tmp);
    }
  }

  in.clear();
  for(int i=0; i<p.second.size(); i++) {
    if(p.second[i]==mapname) {
      int tmp;
      for(tmp=0; tmp<p.mapid.size(); tmp++) {
        if(p.mapid[tmp]==p.first[i]) break;
      }
      in.push_back(tmp);
    }
  }


  starttraining=p.starttraining[id];


  isretina=0;
  for(int i=0; i<p.retinas.size(); i++) {
    if(p.retinas[i]==id) {
      isretina=1;
      break;
    }
  }


  steps.clear();
  for(int i=0; i<p.steps.size(); i++) {
    int tmp=p.steps[i];
    steps.push_back(tmp);
  }

  train.clear();
  for(int i=0; i<p.train[id].size(); i++) {
    param pp=p.train[id][i];
    train.push_back(pp);
  }




  //parse `p.map[id][i]` and create object file
  if(isretina==0) {
    //get first afferent's width/height (AT THE MOMENT LISSOM ONLY ALLOWS `same size` afferents... (it's ok..))

    //puh-leez, assume every layer will have at least 1 input..

    int affw, affh, affsz;
    parse(&affw, "width", p.map[in[0]]);
    parse(&affh, "height", p.map[in[0]]);
    parse(&affsz, "size", p.map[in[0]]);
    if(affsz!=-1) { affw=affsz; affh=affsz; }
    Layer *aff=new Layer;
    aff->w=affw;
    aff->h=affh;


    int width, height, size;
    int scaleareaordensity, settletime, iterations, numinputs;
    float rf, rexc, rinhib, rexcfinal, alphaa, alphaexc, alphainhib, gammaexc, gammainhib;
    float lowerthr, upperthr;

    parse(&width, "width", p.map[id]);
    parse(&height, "height", p.map[id]);
    parse(&size, "size", p.map[id]);
    if(size!=-1) { width=size; height=size; }

    parse(&scaleareaordensity, "scaleareaordensity", p.map[id]);
    parse(&settletime, "settletime", p.map[id]);
    parse(&iterations, "iterations", p.map[id]);

    parse(&numinputs, "numafferent", p.map[id]);
    if(numinputs==-1) numinputs=1;

    parse(&rf, "rf", p.map[id]);
    parse(&rexc, "rexc", p.map[id]);
    parse(&rinhib, "rinhib", p.map[id]);
    parse(&rexcfinal, "rexcfinal", p.map[id]);
    parse(&alphaa, "alphaa", p.map[id]);
    parse(&alphaexc, "alphaexc", p.map[id]);
    parse(&alphainhib, "alphainhib", p.map[id]);
    parse(&gammaexc, "gammaexc", p.map[id]);
    parse(&gammainhib, "gammainhib", p.map[id]);
    parse(&lowerthr, "lowerthr", p.map[id]);
    parse(&upperthr, "upperthr", p.map[id]);


    if(scaleareaordensity==-1) scaleareaordensity=0;
    if(rexc==-1.0) rexc=0.0;
    if(rinhib==-1.0) rinhib=0.0;
    if(rexcfinal==-1.0) rexcfinal=1.13;
    if(alphaa==-1.0) alphaa=0.007;
    if(alphaexc==-1.0) alphaexc=0.00466167;
    if(alphainhib==-1.0) alphainhib=0.0330078;
    if(gammaexc==-1.0) gammaexc=0.9;
    if(gammainhib==-1.0) gammainhib=0.9;
    if(settletime==-1) settletime=9;

    iterationstotrain=iterations;

    if(lowerthr==-1.0) lowerthr=0.1;
    if(upperthr==-1.0) upperthr=0.65;


    layer=new LISSOM(width, height, rf, aff, numinputs, scaleareaordensity, 0, 0, rexc, rinhib, rexcfinal, alphaa, alphaexc, alphainhib, gammaexc, gammainhib, settletime, lowerthr, upperthr);
    delete aff;



    if(action==ACTIONLOADRUN || action==ACTIONLOADTRAIN) {
      if(isretina==0) layer->load( (char *)("out/"+mapname+".lissom").c_str() );
    }



    w=layer->w;
    h=layer->h;

    int maxw=w;
    int maxh=h;



    initrexc=layer->rE;
    finalrexc=layer->rEf;



    //create afferents objects
    inp.clear();
    for(int i=0; i<in.size(); i++) {
      afferentlayer *tmp=new afferentlayer;
      tmp->id=in[i];

      int w_, h_, sz;
      parse(&w_, "width", p.map[in[i]]);
      parse(&h_, "height", p.map[in[i]]);
      parse(&sz, "size", p.map[in[i]]);
      if(sz!=-1) { w_=sz; h_=sz; }

      int pattsperiter;
      parse(&pattsperiter, "patternsperiteration", p.map[in[i]]);
      if(pattsperiter==-1) pattsperiter=-1;


      maxw=max(maxw, w_);
      maxh=max(maxh, h_);


      tmp->layer=new Retina(w_, h_, pattsperiter);


      layer->ConnectAfferent(tmp->layer);

      inp.push_back(tmp);

    }


    obuf=(unsigned char *)malloc(maxw*maxh*sizeof(float));


  } else if(isretina==1) {
    int width, height, size;
    int patternsperiteration;

    parse(&width, "width", p.map[id]);
    parse(&height, "height", p.map[id]);
    parse(&size, "size", p.map[id]);
    parse(&patternsperiteration, "patternsperiteration", p.map[id]);
    if(size!=-1) { width=size; height=size; }
    if(patternsperiteration==-1) patternsperiteration=1;


    w=width;
    h=height;
    obuf=(unsigned char *)malloc(w*h*sizeof(float));


    retina=new Retina(width, height, patternsperiteration);


  }



}




LissomWrapper::~LissomWrapper() {
  free(obuf);
  if(isretina==1) {
    delete retina;
  } else {
    for(int i=0; i<inp.size(); i++) {
      delete inp[i];
    }
    delete layer;
  }


}




void LissomWrapper::getinput(string s) {
  for(int i=0; i<inp.size(); i++) {
    MPI_Status s;

    int err=MPI_Recv(obuf, inp[i]->layer->w*inp[i]->layer->h, MPI_FLOAT, inp[i]->id, 1, MPI_COMM_WORLD, &s);


    if(err!=MPI_SUCCESS) {
      cout<<"Layer[FIRSTSTEP] with id["<<mid<<"] threw this error:"<<err<<"  while receiving data from node #"<<inp[i]->id<<endl;
    }


    inp[i]->layer->setinput(obuf);
  }

}




void LissomWrapper::setoutput(string s) {
  for(int i=0; i<out.size(); i++) {
    MPI_Request r;

    int err=MPI_Isend(obuf, w*h, MPI_FLOAT, out[i], 1, MPI_COMM_WORLD, &r);

    if(err!=MPI_SUCCESS) {
      cout<<s<<" with id["<<mid<<"] produced error:"<<err<<" on sending to process #"<<out[i]<<endl;
    }

  }

}




void LissomWrapper::randomGaussian(int center) {
//if it's a retina,

if(isretina==1) {
 //do the action

  retina->randomGaussian(center);


 //write output to every outcoming stream
  retina->getoutput(obuf); //w


  setoutput("Retina[RANDOMGAUSSIAN]");

}

}



void LissomWrapper::FirstStep() {
// -getinputmap- -process- -write output map to everybody else needing it-

//if it requires >0 inputs, wait for them to arrive

if(isretina==0) {
  getinput("Layer[FIRSTSTEP]");


  layer->FirstStep();


/* //FirstStep shouldn't be broadcasted!
//if >0 outputs, write output to every output map



  if(out.size()!=0) layer->getoutput(obuf);


  setoutput("Layer[FIRSTSTEP]");
*/
}

}




void LissomWrapper::Step() {
if(isretina==0) {
 //Doesn't need inputs (already done on FirstStep)

  layer->Step();


  if(out.size()!=0) layer->getoutput(obuf);

  setoutput("Layer[STEP]");

}

}




void LissomWrapper::AdjustWeights() {
  //No input/output involved here

if(isretina==0) {
  layer->AdjustWeights();

}

}








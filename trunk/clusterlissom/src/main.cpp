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
#include <sstream>
#include <vector>
#include <algorithm>

using namespace std;


#undef SEEK_SET
#undef SEEK_END
#undef SEEK_CUR
#include "mpi.h"


#include "bmp_io.cpp"


#include "config.h"
#include "parser.h"
#include "lissomwrapper.h"

#include "../lissom.h"


int numprocesses, id;




int main(int argc, char **args) {
  MPI_Init(&argc, &args);
  int comm=MPI_COMM_WORLD;
  MPI_Comm_size(comm, &numprocesses); //it will be equal to the number of maps..
  MPI_Comm_rank(comm, &id);

  if(argc!=2) {
    printf("Wrong argument list.\n        Correct usage is %s script.lissom\n\n", args[0]);
    //return 0;
    MPI_Abort(MPI_COMM_WORLD, 0);
  }



  Config c;
  c.open(args[1]);

  Parser p;
  p.open(&c);

  c.close();


  if(numprocesses!=p.mapid.size()) {
    printf("Number of processes must be equal to number of simulated maps (%d)\n\n", p.mapid.size());
    MPI_Abort(MPI_COMM_WORLD, 0);
  }


  //processor with id `id` will handle map # `id`.

  LissomWrapper *lw=new LissomWrapper();
  lw->open(p, id);


/*
lw->randomGaussian();
lw->FirstStep();
lw->Step();
lw->AdjustWeights();
*/



  //Do the selected action
  if(lw->action==ACTIONRUN || lw->action==ACTIONLOADRUN) {
    lw->randomGaussian(1); // <--- TODO: different inputs...
    lw->FirstStep();
    lw->Step();

    //TODO: do something, eg write to images current network output..



    if(lw->isretina==0) lw->layer->getoutput(lw->obuf);
    else lw->retina->getoutput(lw->obuf);

    float *obuf2=(float *)lw->obuf;
    unsigned char *bf=(unsigned char *)malloc(lw->w*lw->h);
    for(int x=0; x<lw->w; x++) {
      for(int y=0; y<lw->h; y++) {
        bf[y*lw->w+x] = (unsigned char)(obuf2[y*lw->w+x]*255.0);
      }
    }

    bmp_24_write((char *)("out/"+lw->mapname+".bmp").c_str(), lw->w, lw->h, bf, bf, bf);

    free(bf);




  } else if(lw->action==ACTIONTRAIN || lw->action==ACTIONLOADTRAIN) {
    //check for biggest starttime+n.iterations

    int maxend=0;

    for(int i=0; i<p.mapid.size(); i++) {
      int tmp;
      parse(&tmp, "iterations", p.map[i]);
      tmp+=p.starttraining[i];

      if(tmp>maxend) maxend=tmp;
    }



    for(int o=0; o<maxend; o++) {
      if(id==0 && o%100==0) printf("--%d\n", o);

      lw->randomGaussian();
      lw->FirstStep();
      lw->Step();


      if(o>=lw->starttraining && lw->isretina==0) {
        int iter=o-lw->starttraining;

        //If the network should be training at this time step
        if(iter<lw->iterationstotrain) {
          lw->AdjustWeights();

          //Check if you should update anything
          for(int i=0; i<lw->steps.size(); i++) {
          if(lw->steps[i]==iter) {

            //Find what you need to update
            for(int j=0; j<lw->train.size(); j++) {
            if(lw->train[j].step==iter) {
              //Update it, it's all into lw->train[j], as a param struct (see lissomwrapper, parser for more information on formatting)

              if(lw->train[j].s=="upperthr") {
                string s=lw->train[j].s2;
                char op;
                float n;

                stringstream st(s);
                st>>op;
                st>>n;

                if(op=='-') n=-n;

                lw->layer->setThresholds(lw->layer->lowerthr, lw->layer->upperthr+n);
              } else if(lw->train[j].s=="lowerthr") {
                string s=lw->train[j].s2;
                char op;
                float n;

                stringstream st(s);
                st>>op;
                st>>n;

                if(op=='-') n=-n;

                lw->layer->setThresholds(lw->layer->lowerthr+n, lw->layer->upperthr);
              } else if(lw->train[j].s=="settletime") {
                if(lw->train[j].s2.size()!=0) { //+/-
                  string s=lw->train[j].s2;
                  char op;
                  int n;

                  stringstream st(s);
                  st>>op;
                  st>>n;

                  if(op=='-') n=-n;

                  lw->layer->setSettleTime(lw->layer->settletime+n);
                } else { //abs
                  lw->layer->setSettleTime(lw->train[j].i);
                }

              } else if(lw->train[j].s=="gammaexc") {
                lw->layer->setGammas(lw->train[j].f, lw->layer->gammaI);
              } else if(lw->train[j].s=="gammainhib") {
                lw->layer->setGammas(lw->layer->gammaE, lw->train[j].f);
              } else if(lw->train[j].s=="alphaa") {
                if(lw->train[j].s2.size()==0) {
                  lw->layer->setAlphaA(lw->train[j].f);
                } else {
                  string s=lw->train[j].s2;
                  char op;
                  float n;

                  stringstream st(s);
                  st>>op;
                  st>>n;

                  lw->layer->setAlphaA(lw->layer->alphaA*n);
                }

              } else if(lw->train[j].s=="alphaexc") {
                if(lw->train[j].s2.size()==0) {
                  lw->layer->setAlphaE(lw->train[j].f);
                } else {
                  string s=lw->train[j].s2;
                  char op;
                  float n;

                  stringstream st(s);
                  st>>op;
                  st>>n;

                  lw->layer->setAlphaE(lw->layer->alphaE*n);
                }

              } else if(lw->train[j].s=="alphainhib") {
                if(lw->train[j].s2.size()==0) {
                  lw->layer->setAlphaI(lw->train[j].f);
                } else {
                  string s=lw->train[j].s2;
                  char op;
                  float n;

                  stringstream st(s);
                  st>>op;
                  st>>n;

                  lw->layer->setAlphaI(lw->layer->alphaI*n);
                }

              } else if(lw->train[j].s=="rexc") {
                string s=lw->train[j].s2;
                char op;
                float n;

                stringstream st(s);
                st>>op;
                st>>n;

                lw->layer->setRe(max(lw->finalrexc, lw->initrexc*n));
              }





            } //end updating properties
            }


            break;
          }
          }


        }
      }





       //synchronize cluster
//       int a, b;
//       MPI_Reduce(&a, &b, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
       MPI_Barrier(MPI_COMM_WORLD);

    }





    //Save Networks to file
    if(lw->isretina==0) lw->layer->save( (char *)("out/"+lw->mapname+".lissom").c_str() );

  }








  delete lw;

  MPI_Finalize();

  return 0;
}



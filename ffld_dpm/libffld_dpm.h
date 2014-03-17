#ifndef LIBFFLD_DPM 
#define LIBFFLD_DPM

#include "SimpleOpt.h"

#include "Intersector.h"
#include "Mixture.h"
#include "Scene.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
using namespace std;

//container for one image's DPM results
class DPM_Results{
    vector<HOGPyramid::Matrix> scores;
    vector<Mixture::Indices> argmaxes;
    vector<vector<vector<Model::Positions> > > positions; 
};


class FFLD_DPM{

public:

    //'threshold' and 'overlap' might be unused, since detect() returns a full score map.
    FFLD_DPM(int padding, int interval, int threshold, int overlap,
             string model_fname);

    ~FFLD_DPM(); //Model's destructor should get called.

    Mixture mixture; //persistant instance of the DPM Model (model_fname), with the parts in FFT space
   
    //TODO: in detect(), do 'int width = image.width(); int height = image.height();' -- this was previously done in main() and passed to detect()
    //TODO: load image and compute HOGPyramid from inside my detect()
    DPM_Results detect(string image_fname); //we just support 1 DPM model at a time for now. Could add a parameter to select among several DPM models.

//the following is for 'internal use,' but making it private would be annoying for debugging printfs outside this class. 
    int padding;
    int interval;

};



#endif


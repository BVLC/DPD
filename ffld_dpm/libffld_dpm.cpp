#include "libffld_dpm.h"

//this doesn't take an 'overlap' or 'threshold' parameter, because the NMS and thresholding are left up to the user.
FFLD_DPM::FFLD_DPM(int input_padding, int input_interval, string model_fname)
{
    //setup class variables
    padding = input_padding;
    interval = input_interval;

    //set up DPM model, which we refer to as a 'mixture' 
    ifstream in(model_fname.c_str(), ios::binary); //open the DPM model file
    if (!in.is_open()) {
        showUsage();
        cerr << "\nInvalid model file " << model << endl;
        return -1;
    }
    in >> mixture; //the FFLD_DPM class's mixture.
   
    //should slow down and make a decision:
    // do I want to run cacheFilters() now -- just FFT the filters once? 
    // or, should I run cacheFilters() once per image?
    // this matters, because we can only recycle cacheFilters results if the input images are all the same size.  
    

}

DPM_Results detect(string image_fname){

    //TODO: load image...
    JPEGImage image(image_fname);

    int width = image.width();
    int height = image.height();

    //hmm, if Patchwork is static, it'd be hard to keep creating new Patchworks for different img sizes.

    if (!Patchwork::Init((width + 15) & ~15, (height + 15) & ~15)) {
        cerr << "\nCould not initialize the Patchwork class" << endl;
        return -1;
    }





}




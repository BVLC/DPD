

#extract DPM part locations, save cropped part bounding boxes in $dpd_scratch
./extract_dpm_parts.sh #calls into ffld_dpm/build/ffld

#DeCAF feature extraction on DPM part locations
./dpd_decaf/run_dpd_decaf_features.sh #DeCAF convnet features on DPM parts

#weak pooling, SVM training, SVM classification
matlab -r "dpd_decaf; exit;" 


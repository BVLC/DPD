#!/bin/sh
dpd_scratch=/media/big_disk/dpd_scratch
#cropped_img_dir=$dpd_scratch/cropped_bird_imgs #DPM part and root part boxes cropped out (to feed into Decaf)
cropped_img_dir=$dpd_scratch/cropped_bird_imgs_ffld 
#decaf_output_dir=$dpd_scratch/decaf_bird_features_forrest
decaf_output_dir=$dpd_scratch/decaf_bird_features_forrest_ffld
num_parts=8 #number of parts in our DPM model. 
max_images=5994 #some parts are missing, but we have at most this many bboxes
subset='train'
mkdir $decaf_output_dir

for subset in 'train' 'test' 
do
    #DeCAF features on DPM-detected parts
    for((partIdx=1; partIdx <= $num_parts; partIdx++)) do #1-indexed part IDs
        #in_img_dir=$cropped_img_dir/${subset}_part_${partIdx}/
        in_img_dir=$cropped_img_dir/${subset}/part_${partIdx}/
        out_feature_mat=$decaf_output_dir/feature_${subset}_part_$partIdx.mat
        python dpd_decaf_features.py $in_img_dir $out_feature_mat $max_images
    done

    #DeCAF features on DPM-detected root filter or ground-truth bounding box
    #in_img_dir=$cropped_img_dir/bbox_${subset}/
    in_img_dir=$cropped_img_dir/${subset}/bbox/
    out_feature_mat=$decaf_output_dir/feature_bbox_${subset}.mat
    python dpd_decaf_features.py $in_img_dir $out_feature_mat $max_images

done



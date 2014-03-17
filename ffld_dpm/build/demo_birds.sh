
for train_or_test in 'train' 'test'
do

    dpd_scratch='/media/big_disk/dpd_scratch'
    in_img_dir=${dpd_scratch}/bird_images_for_ffld/$train_or_test
    in_bbox_dir=${dpd_scratch}/bird_labels_for_ffld/$train_or_test
    out_parts_dir=${dpd_scratch}/cropped_bird_imgs_ffld/$train_or_test
    show_parts_dir=${dpd_scratch}/show_bird_parts_ffld/$train_or_test #temporary -- parts annotated on input imgs

    for curr_img in $in_img_dir/*
    #for curr_img in $in_img_dir/00006.jpg
    do

        #note: --images = output dir for 'images with parts drawn on them'
        #      --out_parts_dir = output dir for 'parts cropped out of the input images'
        
        echo $curr_img

        filename=$(basename "$curr_img") #thanks: stackoverflow.com/questions/965053
        in_bbox_fname=$in_bbox_dir/"${filename%.*}".txt #e.g. 00003.txt
        echo $in_bbox_fname

        #extract parts using DPM
        time ./ffld --out-parts-dir $out_parts_dir --in-bbox-fname $in_bbox_fname --model ../models/dpm_bird_weak.txt --results ./result.txt --images $show_parts_dir --threshold=-2.5 $curr_img

        #echo ./ffld --out-parts-dir $out_parts_dir --in-bbox-fname $in_bbox_fname --model ../models/dpm_bird_weak.txt --results ./result.txt --images $show_parts_dir --threshold=-0.5 $curr_img #for gdb 
    done
done


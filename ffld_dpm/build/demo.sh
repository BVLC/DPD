#time ./ffld --model ../models/bicycle.txt --results ./result.txt --images . --threshold=-0.5 ../bicycle.jpg
#time ./ffld --out-parts-dir ./output_parts --model ../models/bicycle.txt --results ./result.txt --images . --threshold=-0.5 ../bicycle.jpg
time ./ffld --out-parts-dir ./output_parts --model ../models/bicycle_final_voc4.01.txt --results ./result.txt --images . --threshold=-0.5 ../bicycle.jpg

#time ./ffld --model ../models/dpm_bird_weak.txt --results ./result.txt --images . --threshold=-0.5 ../bicycle.jpg

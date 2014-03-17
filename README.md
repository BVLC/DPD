Deformable Part Descriptors (DPD)
===

This code accompanies the ICCV 2013 paper <b>Deformable Part Descriptors for Fine-grained Recognition and Attribute Prediction</b>.


===
<h3>User Configuration</h3>
[TODO: streamline these directory paths for minimal user setup]

in `dpd_set_up.m`:
```Matlab
scratchdir = /scratch  %for KDES features, DPD features, etc

if strcmp(database, 'bird')
    dataset_base = /path/to/CUB200-2011 %you edit this
elseif strcmp(database, 'cub200')
    dataset_base = /path/to/CUB200-2010 %you edit this
elseif strcmp(database, 'human')
    dataset_base = /path/to/berkeley-human-attributes-dataset %you edit this
end
```

===
<h3>Running DPD+DeCAF demos</h3>
Here's how to automatically classify CUB200-2011 birds using Deformable Part Descriptors with DeCAF convolutional features:

TODO: set scratch dir somewhere (env var?)
TODO: add scratch directory argument: dpd_decaf(dpd_scratch)

```Bash

#Option 1: run DPD+DeCAF in one line
./run_dpd_decaf.sh #this script does each step of DPD+DeCAF pipeline

#Option 2: run DPD+DeCAF steps one at a time
./extract_dpm_parts.sh #calls into ffld_dpm/build/ffld
./run_dpd_decaf_features.sh #DeCAF convnet features on DPM parts
matlab
>dpd_decaf; %weak pooling, SVM training, SVM classification

#dpd_decaf is hard-coded for this config: (weak pooling; CUB200_2011; trainAndTest). 
# not too hard to modify, though.
```

===
<h3>Running DPD+KDES demos</h3>
Here's how to automatically classify CUB200-2011 birds using Deformable Part Descriptors with Kernel Descriptors (KDES):

```Matlab
matlab
>run_dpd('bird', 0, 0); %runs all images through each DPD pipeline step, in batch mode

%arguments: run_dpd_kdes('class', 0=weakPooling 1=strongPooling, 0=trainAndTest 1=trainOnly);
%assume: DPM part bounding boxes are already extracted and located in a subdirectory of dpd_scratch
```

===
<h3>Reorganizing directories</h3>
```Shell

move:
dpd_set_up -> dpd_kdes_set_up
thirdparty/ffld_dpm ./ffld_dpm

new: 
dpd/dpd_decaf 
    mv train_test/dpd_decaf.m dpd_decaf/dpd_decaf.m
    mv ./dpd_decaf_features.py dpd_decaf/dpd_decaf_features.py
dpd/dpd_kdes
    mv train_test dpd_kdes
    common/{cksvd*, kdes_cholesky} -> dpd_kdes
dpd/old
    mv dpd/test_realtime old
    mv dpd/test_realtime_cpp old
    mv thirdparty old/thirdparty
    
    mv fgcomp old/fgcomp  #just a script to convert FGcomp labels to pascal format

add:
dpd_decaf_set_up.sh #just point to dpd_scratch directory location



```

#!/bin/sh
cd scripts/

input=SET_PATH_HERE
predictions=SET_PATH_HERE
output=SET_PATH_HERE
sequence=False
packing=True
padding=15
size=100
scale=true
alignCTU=true
reduced_parameters=false

bash run_pipeline.sh $input $predictions $output $sequence $packing $padding $size $scale $alignCTU $reducedParameters

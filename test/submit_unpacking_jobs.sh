#!/bin/sh
cd scripts/

input=SET_PATH_HERE
csv=SET_PATH_HERE
output=SET_PATH_HERE
sequence=True
reducedParameters=True
encoded=False

bash run_unpacking.sh $input $csv $output $sequence $reducedParameters $encoded

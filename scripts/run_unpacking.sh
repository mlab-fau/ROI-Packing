#!/bin/bash

input_path=$1
csv_path=$2
output_path=$3

if [ -z "$7" ];
then
    python run_unpacking.py --input $input_path --csv $csv_path --output $output_path --sequence $4 --reducedParameters $5 --encoded $6
else
    python run_unpacking.py --input $input_path --csv $csv_path --output $output_path --sequence $4 --reducedParameters $5 --encoded $6 --rescaleSize $7
fi

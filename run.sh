#!/bin/bash

# Create output directory.
mkdir -p output

# Itereate all graphs.
for f in ./data/network.txt
do
  # Block sizes from 32 -> 1024 by 32.
  for i in {32..64..32}
    do 
      # echo "Run $f for $i"
      filename=$(basename "$f")
      output="gpu___""$filename""___""$i"
      echo $output
      # command="/usr/local/cuda-8.0/bin/nvprof --metrics all --csv --print-gpu-trace --events all ./graphlets data/network.txt $i > ./output/$output"
      command="nvprof --metrics all --csv --print-gpu-trace --events all ./graphlets data/network.txt 256 > output/$output.txt 2>output/$output.csv"
      echo $command
      $command
  done
done

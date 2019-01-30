#!/bin/bash

if [ $# -gt 0 ]; then
  N=$1
else
  N=4
fi

if [ -f inx.txt ]; then
  rm inx.txt
fi

for i in `seq 1 $N ` ; do
  for j in `seq 1 $N ` ; do
    qij=`echo "scale=2; $RANDOM/32767 - 0.5" | bc` ;
    echo $i $j $qij | tee -a inx.txt
  done
done

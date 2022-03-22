#!/bin/bash

for i in 1 2 3 4 5 6 7 8 9 10 11 12
do
  mpiexec -machinefile ./allnodes -np $i ./l1p 1000000000
done
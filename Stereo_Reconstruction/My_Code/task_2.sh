#!/bin/bash

cd ../LoopAndZhang/build
rm -rf *
cmake ..
make
./main

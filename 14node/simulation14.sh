#!/bin/sh

for eps in 0.4 0.02 0.008;do
		      python3 main_14.py 0.01 $eps
done

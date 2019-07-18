#!/bin/sh

for eps in 0.004 0.0015;do
		      python3 main_14.py 0.01 $eps
done

#!/bin/sh

for eps in 0.4 0.07 0.015;do
		      python3 main_11.py 0.03 $eps
done

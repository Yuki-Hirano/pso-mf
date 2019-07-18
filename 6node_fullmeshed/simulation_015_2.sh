#!/bin/sh

for eps in 0.0013 0.00125 0.0012;do
		      python3 main.py 0.015 $eps
done

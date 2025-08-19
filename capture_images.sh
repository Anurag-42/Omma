#!/bin/bash

for i in $(seq 1 100)
do
  libcamera-jpeg -o image${i}.jpg
  echo "Captured image${i}.jpg"
  sleep 20
done

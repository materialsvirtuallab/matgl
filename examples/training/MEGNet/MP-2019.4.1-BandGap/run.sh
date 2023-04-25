#!/bin/bash

# download the zip file from figshare
echo "wget mp.2019.04.01.json.gz"
wget https://ndownloader.figshare.com/files/15108200
# move the downloaded file to get the correct filename
mv 15108200 mp.2019.04.01.json.gz
# unzip the file
echo "Unzip the file ... "
gunzip mp.2019.04.01.json.gz
# Model fitting
echo "Running training script"
nohup python train.py > log.txt&

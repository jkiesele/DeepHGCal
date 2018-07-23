#!/usr/bin/env python2


import glob
from argparse import ArgumentParser
import os
import subprocess

parser = ArgumentParser('program to create file list of all root files in directory')
parser.add_argument("input_dir")
parser.add_argument("--split", type=float, default=0.1)
args=parser.parse_args()


inputfiles = glob.glob(args.input_dir+"/*.root")
count=0
total=len(inputfiles)

with open(args.input_dir+"/allfiles.txt",'w') as out:
    for f in inputfiles: 
        rootfile=os.path.basename(f)
        out.write(rootfile+'\n')
            
        

with open(args.input_dir+"/train_files.txt",'w') as out:
    for f in inputfiles: 
        rootfile=os.path.basename(f)
        if count< total*(1.-args.split):
            out.write(rootfile+'\n')
            count+=1
       
print(str(count)+' train files') 
trainfiles=count
count=0
with open(args.input_dir+"/test_files.txt",'w') as out:
    for f in inputfiles: 
        rootfile=os.path.basename(f)
        if count >= total*(1.-args.split):
            out.write(rootfile+'\n')
        count+=1

print(str(total-trainfiles)+' test files') 
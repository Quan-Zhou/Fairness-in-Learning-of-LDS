#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --error=/home/zhouqua1/FairNCPOP/err.err 
#SBATCH --output=/home/zhouqua1/FairNCPOP/out.out 
#SBATCH --mem-per-cpu 64G
#SBATCH --time 1-00:00:00 
#SBATCH --partition=amd

import numpy as np
import pandas as pd
import sys
sys.path.append("/home/zhouqua1/FairNCPOP") 
from functions import*

# Set Parameters
level = 1
group = 2
trajectory = [2,2]
sum_traj = sum(trajectory)

# Generate observations of multiple trajectories
data=pd.read_csv('/home/zhouqua1/FairNCPOP/data/FairOutput.csv',header=None).to_numpy()
#Y_orig=data[0:T,:]

Amean=[]
Astd=[]
Bmean=[]
Bstd=[]
for t in range(5,21):
    Am=[]
    Bm=[]
    for r in range(3):
        Y=data[0:t,:]
        Am.append(fairA_timing(Y,trajectory,level))
        Bm.append(fairB_timing(Y,trajectory,level))
    Amean.append(np.mean(Am))
    Astd.append(np.std(Am))
    Bmean.append(np.mean(Bm))
    Bstd.append(np.std(Bm))
A=pd.DataFrame(list(zip(Amean,Astd)))
B=pd.DataFrame(list(zip(Bmean,Bstd)))
A.to_csv ('/home/zhouqua1/FairNCPOP/data/ncpol2sdpaAtime0520.csv', index = False, header=False)
B.to_csv ('/home/zhouqua1/FairNCPOP/data/ncpol2sdpaBtime0520.csv', index = False, header=False)
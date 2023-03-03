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
filename="/home/zhouqua1/FairNCPOP/data/sdpa_export/"

# Generate observations of multiple trajectories
data=pd.read_csv('/home/zhouqua1/FairNCPOP/data/FairOutput.csv',header=None).to_numpy()
#Y_orig=data[0:T,:]

for t in range(5,31):
    Y=data[0:t,:]
    fairA_sparsity(Y,trajectory,level,filename)

z1=[]
z2=[]
for T in range(5,31):
    with open(filename+'sdpa_'+str(T)+'.dat-s') as f:
        lines = f.readlines()
    
    x=int(lines[1][0:lines[1].find('=')])
    y=len(lines)
    z1+=[x**2]
    z2+=[(y-5)*2]
    
df = pd.DataFrame(data={"col1":z1, "col2": z2})
df.to_csv("/home/zhouqua1/FairNCPOP/data/sparsity.csv", sep=',',index=False)
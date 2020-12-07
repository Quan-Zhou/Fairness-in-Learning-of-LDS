!wget -c https://raw.githubusercontent.com/jmarecek/OnlineLDS/master/inputlds.py
import sys
sys.path.append('.')
from inputlds import*

from ncpol2sdpa import*
import numpy as np
import math
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# Set Parameters
level = 1
R=30
met=3
T = 20 
group = 2
trajectory = [2,3]
sum_traj = sum(trajectory)
ff=np.zeros([R,T*met])
g = np.matrix([[0.99,0],[1.0,0.2]])
f_dash = np.matrix([[1.1,0.8]])
proc_noise_std = np.random.rand(1,group)*0.1
obs_noise_std = np.random.rand(1,sum_traj)
inputs = np.zeros(T)
h0=[5,7]


# Generate observations of multiple trajectories 
Y_orig=np.mat(np.empty([T,sum_traj]))
j=0
for s in range(group):
    for i in range(trajectory[s]):
        ds1 = dynamical_system(g,np.zeros((2,1)),f_dash,np.zeros((1,1)),
        process_noise='gaussian',
        observation_noise='gaussian', 
        process_noise_std=proc_noise_std[0,s], 
        observation_noise_std=obs_noise_std[0,j])
        h1=np.ones(ds1.d)*h0[s]
        ds1.solve(h0=h1,inputs=inputs,T=T)
        Y_orig[:,j]=np.vstack(ds1.outputs)
        j+=1
            
# Write Y_orig data to csv files
pd.DataFrame(Y_orig).to_csv (r'FairOutput.csv', index = False, header=False)        

# Decision Variables
G = generate_operators("G", n_vars=1, hermitian=True, commutative=False)[0]
Fdash = generate_operators("Fdash", n_vars=1, hermitian=True, commutative=False)[0]
z = generate_operators("z", n_vars=1, hermitian=True, commutative=True)[0]
m = generate_operators("m", n_vars=T+1, hermitian=True, commutative=False)
q = generate_operators("q", n_vars=T, hermitian=True, commutative=False)
p = generate_operators("p", n_vars=T, hermitian=True, commutative=True)
f = generate_operators("f", n_vars=T, hermitian=True, commutative=True)
# Constraints
ine1 = [f[i] - Fdash*m[i+1] - p[i] for i in range(T)]
ine2 = [-f[i] + Fdash*m[i+1] + p[i] for i in range(T)]
ine3 = [m[i+1] - G*m[i] - q[i] for i in range(T)]
ine4 = [-m[i+1] + G*m[i] + q[i] for i in range(T)]
ines_unfair = ine1+ine2+ine3+ine4
# Objective
obj_A = z + 1*sum(p[i]**2 for i in range(T)) # 1 is optimal for level 1
obj_B = z + 3*sum(p[i]**2 for i in range(T)) # 3 is optimal for level 1


# Solve
r=0
while r < R:
    # make varying length of multiple trajectories
    start=[0]+random.sample(range(0,10),trajectory[0]-1)+[0]+random.sample(range(0,5),trajectory[1]-1)
    end=random.sample(range(T-10,T),trajectory[0]-1)+[T]+random.sample(range(T-5,T),trajectory[1]-1)+[T]
    length=[end[i]-start[i] for i in range(sum_traj)]
    ind=np.zeros([T,sum_traj])
    for j in range(sum_traj):
        ind[start[j]:end[j],j]=1
    Y = np.multiply(Y_orig,ind)
    
    # Constraints
    max1 = [z-1/trajectory[0]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in range(start[j],end[j])) for j in range(0,trajectory[0]))]
    max2 = [z-1/trajectory[1]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in range(start[j],end[j])) for j in range(trajectory[0],sum_traj))]
    max3 = [z-(Y[t,j]-f[t])**2 for j in range(sum_traj) for t in range(start[j],end[j]) ]
    ines_A = ine1+ine2+ine3+ine4+max1+max2
    ines_B = ine1+ine2+ine3+ine4+max3
    
    # Objective
    obj_unfair = sum(sum((Y[t,j]-f[t])**2 for t in range(start[j],end[j])) for j in range(sum_traj)) + 5*sum(p[i]**2 for i in range(T)) # 5 is optimal for level 1
    
    # Solve the fair NCPO A
    sdp_A = SdpRelaxation(variables = flatten([G,Fdash,z,f,p,m,q]), verbose = 2)
    sdp_A.get_relaxation(level, objective=obj_A, inequalities=ines_A)
    sdp_A.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:\\Users\\zhouq\\Documents\\sdpa7-windows\\sdpa.exe"})
    #print(sdp_A.primal, sdp_A.dual, sdp_A.status)
    
    # Solve the fair NCPO B
    sdp_B = SdpRelaxation(variables = flatten([G,Fdash,z,f,p,m,q]), verbose = 2)
    sdp_B.get_relaxation(level, objective=obj_B, inequalities=ines_B)
    #sdp.get_relaxation(level, objective=obj, inequalities=ines,substitutions=subs)
    sdp_B.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:\\Users\\zhouq\\Documents\\sdpa7-windows\\sdpa.exe"})
    #print(sdp_B.primal, sdp_B.dual, sdp_B.status)
    
    # Solve the unfair NCPO
    sdp_unfair = SdpRelaxation(variables = flatten([G,Fdash,f,p,m,q]), verbose = 2)
    sdp_unfair.get_relaxation(level, objective=obj_unfair, inequalities=ines_unfair)
    sdp_unfair.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:\\Users\\zhouq\\Documents\\sdpa7-windows\\sdpa.exe"})
    #print(sdp_unfair.primal, sdp_unfair.dual, sdp_unfair.status)

    # extract forecast
    # make sure all feasible
    if(sdp_A.status != 'infeasible' and sdp_B.status != 'infeasible' and sdp_unfair.status != 'infeasible'):
        ff_A=[sdp_A[f[i]] for i in range(T)]
        ff_B=[sdp_B[f[i]] for i in range(T)]
        unff=[sdp_unfair[f[i]] for i in range(T)]
        ff[r,:]=flatten([ff_A,ff_B,unff])
        r+=1
   

# Write to files
ff=pd.DataFrame(ff)
ff.to_csv('CompareOutput_ff.csv',index=False,header=False)
outputs=open('CompareOutput_ff.txt','w')
print("proc_noise_std:",proc_noise_std,file=outputs)
print("obs_noise_std:",obs_noise_std,file=outputs)
print("Y_orig:",Y_orig,file=outputs)
outputs.close()


# Plot
#ff=pd.read_csv('CompareOutput_ff.csv',header=None)
col1=ff.to_numpy().flatten() #values
col2=[t for t in range(T)]*met*R #time
col3=flatten([[m]*T for m in range(met)]*R) #type
df=pd.DataFrame(np.vstack((col1,col2,col3)).T,columns=['Forecast', 'Time', 'raw_type'])
df["Model type"] = df["raw_type"].astype("category")
df["Model type"].cat.categories = ["Subgroup-Fair", "Instant-Fair", "Unfair"]
df["Time"]=df["Time"].astype(int) 
df=df.drop(columns=['raw_type'])

sns.set(style="white",palette=["cyan","chartreuse","red"])
ax=sns.lineplot(x="Time",y="Forecast",hue="Model type",data=df,ci="sd")
ax=plt.xticks([i for i in range(T)])


ax=plt.plot(range(T),Y_orig[:,0],'y--',color="lightgrey")
ax=plt.plot(range(T),Y_orig[:,1],'y--',color="lightgrey")
ax=plt.plot(range(T),Y_orig[:,2],'y:',color="lightgrey")
ax=plt.plot(range(T),Y_orig[:,3],'y:',color="lightgrey")
ax=plt.plot(range(T),Y_orig[:,4],'y:',color="lightgrey")

plt.savefig('Overview.pdf', bbox_inches='tight') 



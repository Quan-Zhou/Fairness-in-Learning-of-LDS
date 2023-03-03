import sys
sys.path.append("/home/zhouqua1/FairNCPOP") 
from inputlds import*
from functions import*
from ncpol2sdpa import*
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# Set parameters
T = 20 
trajectory = [2,3]
g = np.matrix([[0.99,0],[1.0,0.2]])
f_dash = np.matrix([[1.1,0.8]])
h0=[5,7]

level=1
met=3
repeat=10

# data generation
Y_orig=traj_generation(g,f_dash,trajectory,h0,T)

#pd.DataFrame(Y_orig).to_csv ('/home/zhouqua1/FairNCPOP/data/FairOutput.csv', index = False, header=False)
#data=pd.read_csv('/home/zhouqua1/FairNCPOP/data/FairOutput.csv',header=None).to_numpy()
#Y_orig=data[0:T,:]

ff=SimCom(Y_orig,trajectory,level,repeat)

# Plot
col1=ff.flatten()
col2=[t for t in range(T)]*met*repeat #time
col3=flatten([[m]*T for m in range(met)]*repeat) #type
df=pd.DataFrame(np.vstack((col1,col2,col3)).T,columns=['Forecast', 'Time', "Model type"])
df["Model type"]=df["Model type"].astype("category")
df['Model type']=df['Model type'].cat.rename_categories({0:"Subgroup-Fair",1:"Instant-Fair",2:"Unfair"})
#df["Model type"].cat.categories = ["Subgroup-Fair", "Instant-Fair", "Unfair"]
df["Time"]=df["Time"].astype(int) 
#df=df.drop(columns=['raw_type'])

sns.set(style="white",palette=["cyan","chartreuse","red"])
ax=sns.lineplot(x="Time",y="Forecast",hue="Model type",data=df,errorbar='sd')
ax=plt.xticks([i for i in range(T)])

for i in range(trajectory[0]):
    ax=plt.plot(range(T),Y_orig[:,i],'y--',color="lightgrey")
for i in range(trajectory[1]):
    ax=plt.plot(range(T),Y_orig[:,trajectory[0]+i],'y:',color="lightgrey")

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.legend(fontsize=14)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.xlabel("Time", fontsize=15)
plt.ylabel("Forecast", fontsize=15)

plt.savefig('/home/zhouqua1/FairNCPOP/plots/F1.pdf', bbox_inches='tight') 

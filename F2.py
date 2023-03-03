import sys
sys.path.append("/home/zhouqua1/FairNCPOP") 
sys.path.append("/home/zhouqua1/FairNCPOP/data") 
from functions import*
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# data =====================
T = 20 
trajectory = [2,2]
g = np.matrix([[0.99,0],[1.0,0.2]])
f_dash = np.matrix([[1.1,0.8]])
h0=[5,7]

data=pd.DataFrame(traj_generation(g,f_dash,trajectory,h0,30))
pd.DataFrame(data).to_csv ('/home/zhouqua1/FairNCPOP/data/FairOutput.csv', index = False, header=True)
#=====================================

# Set Parameters
repeat=10
met=3
level = 1
T = 20
group = 2
trajectory = [2,2]

beltaRange=np.arange(0.5, 0.91, 0.1)
beltaRange=[round(i,1) for i in beltaRange]

data=pd.read_csv('/home/zhouqua1/FairNCPOP/data/FairOutput.csv',header=0).to_numpy()
Y=data[0:T,:]

fit_df = pd.DataFrame(columns=['nrmse', 'group', 'trajectory','model type','belta'])
for belta in beltaRange:
    fit_tmp=Simcom_bias(Y,trajectory,level,repeat,belta)
    fit_df=fit_df.append(fit_tmp, ignore_index=True)

fit_df['belta']=fit_df['belta'].astype(str) 
fit_df.to_csv('/home/zhouqua1/FairNCPOP/data/F2.csv', index = False, header=False)

# plot
ax1=plt.subplot(1,2,1)
#fig, ax = plt.subplots()
fit_0=fit_df[fit_df['group']==0]
sns.boxplot(x="belta", y="nrmse", hue="model type", data=fit_0,palette=["cyan","lightgreen","tomato"]).set(xlabel='Beta^(d)',title='Disadvantaged Subgroup',ylabel='nrmse^(d)')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
plt.ylim(0,2)

plt.legend(fontsize=13)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title('Disadvantaged Subgroup',fontsize=12)
plt.xlabel('Beta^(d)',fontsize=12)
plt.ylabel("nrmse^(d)", fontsize=12)

ax2=plt.subplot(1,2,2)
#fig, ax = plt.subplots()
fit_1=fit_df[fit_df['group']==1]
sns.boxplot(x="belta", y="nrmse", hue="model type", data=fit_1,palette=["blue","lime","red"]).set(xlabel='Beta^(d)',title='Advantaged Subgroup',ylabel='nrmse^(a)')#.set_title('Disadvantage Group')
ax2.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
plt.ylim(0,2)

plt.legend(fontsize=13)
plt.xticks(fontsize=10)
#plt.yticks(fontsize=10)
plt.title('Advantaged Subgroup',fontsize=12)
plt.xlabel('Beta^(d)',fontsize=12)
plt.ylabel("nrmse^(a)", fontsize=12)
plt.yticks([])

plt.subplots_adjust(wspace=0.15, hspace=0)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.savefig('/home/zhouqua1/FairNCPOP/plots/F2.pdf', bbox_inches='tight') 
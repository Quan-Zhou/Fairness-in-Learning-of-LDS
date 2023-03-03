import sys
sys.path.append("/home/zhouqua1/FairNCPOP") 
from inputlds import*
from ncpol2sdpa import*
import numpy as np
import pandas as pd
from math import sqrt
import random

def data_generation(g,f_dash,proc_noise_std,obs_noise_std,initial_state,T):
# Generate Dynamic System ds1
    dim=len(g)
    ds1 = dynamical_system(g,np.zeros((dim,1)),f_dash,np.zeros((1,1)),
          process_noise='gaussian',
          observation_noise='gaussian', 
          process_noise_std=proc_noise_std, 
          observation_noise_std=obs_noise_std)
    h0= np.ones(ds1.d)*initial_state
    inputs = np.zeros(T)
    ds1.solve(h0=h0, inputs=inputs, T=T)    
    return np.asarray(ds1.outputs).reshape(-1).tolist()

def traj_generation(g,f_dash,trajectory,h0,T):
# Generate observations of multiple trajectories 
    group=len(trajectory)
    sum_traj = sum(trajectory)
    Y=np.mat(np.empty([T,sum_traj]))
    proc_noise_std = np.random.rand(1,group)*0.1
    obs_noise_std = np.random.rand(1,sum_traj)
    j=0
    for s in range(group):
        for i in range(trajectory[s]):
            outputs=data_generation(g,f_dash,proc_noise_std[0,s],obs_noise_std[0,j],h0[s],T)
            Y[:,j]=np.vstack(outputs)
            j+=1
    return Y

def SimCom(Y_orig,trajectory,level,repeat):

    T,sum_traj=Y_orig.shape
    ff=np.zeros([repeat,T*3])
    
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
    obj_A = z + 1*sum(p[i]**2 for i in range(T))+ 0.01*sum(q[i]**2 for i in range(T)) # 1 is optimal for level 1
    obj_B = z + 3*sum(p[i]**2 for i in range(T))+ 0.01*sum(q[i]**2 for i in range(T)) # 3 is optimal for level 1

    # Solve
    r=0
    while r < repeat:
        # make varying length of multiple trajectories
        belta=0.1
        length=int(T*(1-0.1))
        trajlist=np.zeros([length,sum_traj])
        ind=np.zeros([T,sum_traj])
        for j in range(sum_traj):
            list_tmp=random.sample(range(T), int(T*(1-belta)))
            trajlist[:,j]=list_tmp
            ind[list_tmp,j]=1
        Y = np.multiply(Y_orig,ind)

        # Constraints
        max1 = [z-1/trajectory[0]*sum(1/length*sum((Y[int(t),j]-f[int(t)])**2 for t in trajlist[:,j]) for j in range(0,trajectory[0]))]
        max2 = [z-1/trajectory[1]*sum(1/length*sum((Y[int(t),j]-f[int(t)])**2 for t in trajlist[:,j]) for j in range(trajectory[0],sum_traj))]
        max3 = [z-(Y[int(t),j]-f[int(t)])**2 for j in range(sum_traj) for t in trajlist[:,j]]
        ines_A = ine1+ine2+ine3+ine4+max1+max2
        ines_B = ine1+ine2+ine3+ine4+max3

        # Objective
        obj_unfair = sum(sum((Y[int(t),j]-f[int(t)])**2 for t in trajlist[:,j]) for j in range(sum_traj)) + 5*sum(p[i]**2 for i in range(T)) + 0.01*sum(q[i]**2 for i in range(T)) # 5 is optimal for level 1

        # Solve the fair NCPO A
        sdp_A = SdpRelaxation(variables = flatten([G,Fdash,z,f,p,m,q]), verbose = 0)
        sdp_A.get_relaxation(level, objective=obj_A, inequalities=ines_A)
        sdp_A.solve(solver='mosek')
        #print(sdp_A.primal, sdp_A.dual, sdp_A.status)

        # Solve the fair NCPO B
        sdp_B = SdpRelaxation(variables = flatten([G,Fdash,z,f,p,m,q]), verbose = 0)
        sdp_B.get_relaxation(level, objective=obj_B, inequalities=ines_B)
        #sdp.get_relaxation(level, objective=obj, inequalities=ines,substitutions=subs)
        sdp_B.solve(solver='mosek')
        
        # Solve the unfair NCPO
        sdp_unfair = SdpRelaxation(variables = flatten([G,Fdash,f,p,m,q]), verbose = 0)
        sdp_unfair.get_relaxation(level, objective=obj_unfair, inequalities=ines_unfair)
        sdp_unfair.solve(solver='mosek')
        
        # extract forecast
        # make sure all feasible
        if(sdp_A.status != 'infeasible' and sdp_B.status != 'infeasible' and sdp_unfair.status != 'infeasible'):
            ff_A=[sdp_A[f[i]] for i in range(T)]
            ff_B=[sdp_B[f[i]] for i in range(T)]
            unff=[sdp_unfair[f[i]] for i in range(T)]
            ff[r,:]=flatten([ff_A,ff_B,unff])
            r+=1
    return ff

def Simcom_bias(Y_orig,trajectory,level,repeat,belta):
    
    T=Y_orig.shape[0]
    sum_traj=sum(trajectory)
    Y=Y_orig[:,0:sum_traj]

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
    obj_A = z + 1*sum(p[i]**2 for i in range(T))+ 0.01*sum(q[i]**2 for i in range(T)) # 1 is optimal for level 1
    obj_B = z + 1*sum(p[i]**2 for i in range(T))+ 0.01*sum(q[i]**2 for i in range(T)) # 3 is optimal for level 1
    
    r=0
    while r < repeat: 
    
        traj1list=random.sample(range(T), int(T*(1-belta)))
        traj1=list(set([*range(T)]) - set(traj1list))
        traj2list=random.sample(traj1, int(T*(1-belta)))
        traj2=list(set([*range(T)]) - set(traj2list))
        gro1Remain=pd.DataFrame(list(zip(traj1,traj2)))
        length=[gro1Remain.shape[0]]*trajectory[0]+[T]*trajectory[1]

        # Constraints
        max1 = [z-1/trajectory[0]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in gro1Remain[j]) for j in range(0,trajectory[0])) ]
        max2 = [z-1/trajectory[1]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in range(T)) for j in range(trajectory[0],sum_traj))]
        max3 = [z-(Y[t,j]-f[t])**2 for j in range(0,trajectory[0]) for t in gro1Remain[j]]
        max4 = [z-(Y[t,j]-f[t])**2 for j in range(trajectory[0],sum_traj) for t in range(T)]
        ines_A = ine1+ine2+ine3+ine4+max1+max2
        ines_B = ine1+ine2+ine3+ine4+max3+max4

        # Objective
        sum_error=[(Y[t,j]-f[t])**2 for j in range(trajectory[0]) for t in gro1Remain[j] ] + [(Y[t,j]-f[t])**2 for j in range(trajectory[0],sum_traj) for t in range(T)]
        obj_unfair = sum(sum_error) + 1*sum(p[i]**2 for i in range(T)) + 0.01*sum(q[i]**2 for i in range(T)) # 5 is optimal for level 1

        # Solve the fair NCPO A
        sdp_A = SdpRelaxation(variables = flatten([G,Fdash,z,f,p,m,q]), verbose = 0)
        sdp_A.get_relaxation(level, objective=obj_A, inequalities=ines_A)
        sdp_A.solve(solver='mosek')
        if (sdp_A.status == 'infeasible'): 
            continue

        # Solve the fair NCPO B
        sdp_B = SdpRelaxation(variables = flatten([G,Fdash,z,f,p,m,q]), verbose = 0)
        sdp_B.get_relaxation(level, objective=obj_B, inequalities=ines_B)
        sdp_B.solve(solver='mosek')
        if (sdp_B.status == 'infeasible'): 
            continue

        # Solve the unfair NCPO
        sdp_unfair = SdpRelaxation(variables = flatten([G,Fdash,f,p,m,q]), verbose = 0)
        sdp_unfair.get_relaxation(level, objective=obj_unfair, inequalities=ines_unfair)
        sdp_unfair.solve(solver='mosek')
        if (sdp_unfair.status == 'infeasible'): 
            continue

        # Calculate nrmse         
        mean_j=[np.mean([Y[t,j] for t in gro1Remain[j] ]) for j in range(trajectory[0])]+[np.mean([Y[t,j] for t in range(T)]) for j in range(trajectory[0],sum_traj)]
        error_gro1=[sum( (Y[t,j]-f[t])**2 for t in gro1Remain[j] ) for j in range(trajectory[0])]
        error_gro2=[sum( (Y[t,j]-f[t])**2 for t in range(T) ) for j in range(trajectory[0],sum_traj)]
        nrmse_A_gro1 = [sqrt( sdp_A[error_gro1[j]] / sum((Y[t,j]-mean_j[j])**2 for t in gro1Remain[j]) /length[j] ) for j in range(trajectory[0])]
        nrmse_A_gro2 = [sqrt( sdp_A[error_gro2[j-trajectory[0]]] / sum((Y[t,j]-mean_j[j])**2 for t in range(T)) /length[j]) for j in range(trajectory[0],sum_traj)]
        nrmse_B_gro1 = [sqrt( sdp_B[error_gro1[j]] / sum((Y[t,j]-mean_j[j])**2 for t in gro1Remain[j]) /length[j]) for j in range(trajectory[0])]
        nrmse_B_gro2 = [sqrt( sdp_B[error_gro2[j-trajectory[0]]] / sum((Y[t,j]-mean_j[j])**2 for t in range(T)) /length[j]) for j in range(trajectory[0],sum_traj)]
        nrmse_U_gro1 = [sqrt( sdp_unfair[error_gro1[j]] / sum((Y[t,j]-mean_j[j])**2 for t in gro1Remain[j]) /length[j]) for j in range(trajectory[0])]
        nrmse_U_gro2 = [sqrt( sdp_unfair[error_gro2[j-trajectory[0]]] / sum((Y[t,j]-mean_j[j])**2 for t in range(T)) /length[j]) for j in range(trajectory[0],sum_traj)]

        nrmse_ABU=nrmse_A_gro1+nrmse_A_gro2+nrmse_B_gro1+nrmse_B_gro2+nrmse_U_gro1+nrmse_U_gro2
        group_ABU=flatten([[0]*trajectory[0]+[1]*trajectory[1]]*3)
        traj_ABU=flatten([*range(sum_traj)]*3)
        type_ABU=['Subgroup-Fair']*sum_traj+['Instant-Fair']*sum_traj+['Unfair']*sum_traj
        belta_ABU=[belta]*3*sum_traj
        r_df={'nrmse':nrmse_ABU,'group':group_ABU,'trajectory':traj_ABU,'model type':type_ABU,'belta':belta_ABU}
        r_df=pd.DataFrame(r_df)
        
        r+=1

    return r_df

import time
def fairA_timing(Y,trajectory,level):

    T=Y.shape[0]
    sum_traj=sum(trajectory)
    
    start=[0]+random.sample(range(0,3),trajectory[0]-1)+[0]+random.sample(range(0,2),trajectory[1]-1)
    end=random.sample(range(T-3,T),trajectory[0]-1)+[T]+random.sample(range(T-2,T),trajectory[1]-1)+[T]
    length=[end[i]-start[i] for i in range(sum_traj)]

    # Decision Variables
    G = generate_operators("G", n_vars=1, hermitian=False, commutative=False)[0]
    Fdash = generate_operators("Fdash", n_vars=1, hermitian=False, commutative=False)[0]
    z = generate_operators("z", n_vars=1, hermitian=False, commutative=False)[0]
    m = generate_operators("m", n_vars=T+1, hermitian=False, commutative=False)
    q = generate_operators("q", n_vars=T, hermitian=False, commutative=False)
    p = generate_operators("p", n_vars=T, hermitian=False, commutative=False)
    f = generate_operators("f", n_vars=T, hermitian=False, commutative=False)

    # Constraints
    ine1 = [f[i] - Fdash*m[i+1] - p[i] for i in range(T)]
    ine2 = [-f[i] + Fdash*m[i+1] + p[i] for i in range(T)]
    ine3 = [m[i+1] - G*m[i] - q[i] for i in range(T)]
    ine4 = [-m[i+1] + G*m[i] + q[i] for i in range(T)]
    max1 = [z-1/trajectory[0]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in range(start[j],end[j])) for j in range(0,trajectory[0]))]
    max2 = [z-1/trajectory[1]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in range(start[j],end[j])) for j in range(trajectory[0],sum_traj))]
    #max3 = [z-(Y[t,j]-f[t])**2 for j in range(sum_traj) for t in range(start[j],end[j]) ]

    ines_A = ine1+ine2+ine3+ine4+max1+max2
    #ines_B = ine1+ine2+ine3+ine4+max3

    # Objective
    obj_A = z + 1*sum(p[i]**2 for i in range(T)) + 1*sum(q[i]**2 for i in range(T))

    # Solve the fair NCPO A
    time_start = time.time()
    sdp_A = SdpRelaxation(variables = flatten([G,Fdash,z,f,p,m,q]))
    sdp_A.get_relaxation(level, objective=obj_A, inequalities=ines_A)
    sdp_A.solve(solver='mosek')
    time_end = time.time()
    return time_end-time_start


def fairB_timing(Y,trajectory,level):

    T=Y.shape[0]
    sum_traj=sum(trajectory)
    
    start=[0]+random.sample(range(0,3),trajectory[0]-1)+[0]+random.sample(range(0,2),trajectory[1]-1)
    end=random.sample(range(T-3,T),trajectory[0]-1)+[T]+random.sample(range(T-2,T),trajectory[1]-1)+[T]
    length=[end[i]-start[i] for i in range(sum_traj)]

    # Decision Variables
    G = generate_operators("G", n_vars=1, hermitian=False, commutative=False)[0]
    Fdash = generate_operators("Fdash", n_vars=1, hermitian=False, commutative=False)[0]
    z = generate_operators("z", n_vars=1, hermitian=False, commutative=False)[0]
    m = generate_operators("m", n_vars=T+1, hermitian=False, commutative=False)
    q = generate_operators("q", n_vars=T, hermitian=False, commutative=False)
    p = generate_operators("p", n_vars=T, hermitian=False, commutative=False)
    f = generate_operators("f", n_vars=T, hermitian=False, commutative=False)

# Constraints
    ine1 = [f[i] - Fdash*m[i+1] - p[i] for i in range(T)]
    ine2 = [-f[i] + Fdash*m[i+1] + p[i] for i in range(T)]
    ine3 = [m[i+1] - G*m[i] - q[i] for i in range(T)]
    ine4 = [-m[i+1] + G*m[i] + q[i] for i in range(T)]
    #max1 = [z-1/trajectory[0]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in range(start[j],end[j])) for j in range(0,trajectory[0]))]
    #max2 = [z-1/trajectory[1]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in range(start[j],end[j])) for j in range(trajectory[0],sum_traj))]
    max3 = [z-(Y[t,j]-f[t])**2 for j in range(sum_traj) for t in range(start[j],end[j])]

    #ines_A = ine1+ine2+ine3+ine4+max1+max2
    ines_B = ine1+ine2+ine3+ine4+max3

# Objective
    obj_B = z + 3*sum(p[i]**2 for i in range(T))+ 1*sum(q[i]**2 for i in range(T))

# Solve the fair NCPO B
    time_start = time.time()
    sdp_B = SdpRelaxation(variables = flatten([G,Fdash,z,f,p,m,q]))
    sdp_B.get_relaxation(level, objective=obj_B, inequalities=ines_B)
    sdp_B.solve(solver='mosek')
    time_end = time.time()
    return time_end-time_start

def fairA_sparsity(Y,trajectory,level,filename):

    T=Y.shape[0]
    sum_traj=sum(trajectory)
    
    start=[0]+random.sample(range(0,3),trajectory[0]-1)+[0]+random.sample(range(0,2),trajectory[1]-1)
    end=random.sample(range(T-3,T),trajectory[0]-1)+[T]+random.sample(range(T-2,T),trajectory[1]-1)+[T]
    length=[end[i]-start[i] for i in range(sum_traj)]

    # Decision Variables
    G = generate_operators("G", n_vars=1, hermitian=False, commutative=False)[0]
    Fdash = generate_operators("Fdash", n_vars=1, hermitian=False, commutative=False)[0]
    z = generate_operators("z", n_vars=1, hermitian=False, commutative=False)[0]
    m = generate_operators("m", n_vars=T+1, hermitian=False, commutative=False)
    q = generate_operators("q", n_vars=T, hermitian=False, commutative=False)
    p = generate_operators("p", n_vars=T, hermitian=False, commutative=False)
    f = generate_operators("f", n_vars=T, hermitian=False, commutative=False)

    # Constraints
    ine1 = [f[i] - Fdash*m[i+1] - p[i] for i in range(T)]
    ine2 = [-f[i] + Fdash*m[i+1] + p[i] for i in range(T)]
    ine3 = [m[i+1] - G*m[i] - q[i] for i in range(T)]
    ine4 = [-m[i+1] + G*m[i] + q[i] for i in range(T)]
    max1 = [z-1/trajectory[0]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in range(start[j],end[j])) for j in range(0,trajectory[0]))]
    max2 = [z-1/trajectory[1]*sum(1/length[j]*sum((Y[t,j]-f[t])**2 for t in range(start[j],end[j])) for j in range(trajectory[0],sum_traj))]
    #max3 = [z-(Y[t,j]-f[t])**2 for j in range(sum_traj) for t in range(start[j],end[j]) ]

    ines_A = ine1+ine2+ine3+ine4+max1+max2
    #ines_B = ine1+ine2+ine3+ine4+max3

    # Objective
    obj_A = z + 1*sum(p[i]**2 for i in range(T)) + 1*sum(q[i]**2 for i in range(T))

    # Solve the fair NCPO A
    sdp_A = SdpRelaxation(variables = flatten([G,Fdash,z,f,p,m,q]))
    sdp_A.get_relaxation(level, objective=obj_A, inequalities=ines_A)
    sdp_A.write_to_file(filename+'sdpa_'+str(T)+'.dat-s')
    return 

def split_index(index):
    length = len(index)
    # define the ratios 8:2
    train_len = int(length * 0.8)

    # split the dataframe
    #idx = [*range(length)]
    random.shuffle(index)  # shuffle the index
    I_train = index[:train_len]
    I_test = index[train_len:length]

    return [I_train,I_test]

def compas_pred(index):
    Sindex=Compas(df=dataframe.loc[index])
    Sindex_pred=Sindex.copy(deepcopy=True)
    Sindex_pred.scores=np.array([[1-i/10] for i in dataframe.loc[index,'decile_score'].tolist()])
    #Sindex_pred.labels = np.where(Sindex_pred.scores >= thresh,Sindex_pred.favorable_label,Sindex_pred.unfavorable_label)
    #y_pred = np.zeros_like(Sindex.labels)
    #y_pred[dataframe.loc[index,'decile_score']>thresh]=Sindex.unfavorable_label
    #Sindex_pred.labels=y_pred
    return [Sindex,Sindex_pred]

def preprocess(filepath,nrows,column_names):
    dataframe = pd.read_csv(filepath, index_col='id',nrows=nrows)
    dataframe=dataframe[(dataframe["race"]=='African-American')|(dataframe["race"]=='Caucasian')]
    dataframe[(dataframe.days_b_screening_arrest <= 30)
                & (dataframe.days_b_screening_arrest >= -30)
                & (dataframe.is_recid != -1)
                & (dataframe.c_charge_degree != 'O')
                & (dataframe.score_text != 'N/A')]
    dataframe=dataframe[column_names] #[features+labels]
    dataframe=dataframe.dropna(axis=0, how='any',subset=None,inplace=False) #,thresh=None
    
    dataframe['priors_total_count']=(dataframe['juv_fel_count']+dataframe['juv_misd_count']+dataframe['priors_count'])/3 #+dataframe['juv_other_count']
    
    # base rate: 1-P(is_recid=1|S=s)
    base0=(1-dataframe[(dataframe["race"]=='Caucasian') & (dataframe["is_recid"]==1)].shape[0]/dataframe[dataframe["race"]=='Caucasian'].shape[0])*100
    base1=(1-dataframe[(dataframe["race"]!='Caucasian') & (dataframe["is_recid"]==1)].shape[0]/ dataframe[dataframe["race"]!='Caucasian'].shape[0])*100
    return dataframe,[base0,base1]

def training_process(dataframe,Itrain,method): #Itrain,method
    Dtrain=dataframe.loc[Itrain]
    I0_train=Dtrain[Dtrain['race']=='Caucasian'].index
    I1_train=Dtrain[Dtrain['race']!='Caucasian'].index
    level = 1
    
    if method=='subgroup-fair':
        # Decision Variables
        a = generate_operators("a", n_vars=2, hermitian=True, commutative=False) # age < 25
        c = generate_operators("c", n_vars=2, hermitian=True, commutative=False) # total number of previous convictions
        d = generate_operators("d", n_vars=2, hermitian=True, commutative=False)
        e = generate_operators("e", n_vars=2, hermitian=True, commutative=False)
        z = generate_operators("z", n_vars=3, hermitian=True, commutative=True)

        # Constraints
        ine1 = [z[0]+Dtrain.loc[i,"is_recid"] - a[0]*int(Dtrain.loc[i,'age']<25) - c[0]*Dtrain.loc[i,'priors_total_count'] - d[0]*Dtrain.loc[i,'decile_score'] + e[0] for i in I0_train]
        ine2 = [z[0]-Dtrain.loc[i,"is_recid"] + a[0]*int(Dtrain.loc[i,'age']<25) + c[0]*Dtrain.loc[i,'priors_total_count'] + d[0]*Dtrain.loc[i,'decile_score'] + e[0] for i in I0_train]
        ine3 = [z[0]+Dtrain.loc[i,"is_recid"] - a[1]*int(Dtrain.loc[i,'age']<25) - c[1]*Dtrain.loc[i,'priors_total_count'] - d[1]*Dtrain.loc[i,'decile_score'] + e[1] for i in I1_train]
        ine4 = [z[0]-Dtrain.loc[i,"is_recid"] + a[1]*int(Dtrain.loc[i,'age']<25) + c[1]*Dtrain.loc[i,'priors_total_count'] + d[1]*Dtrain.loc[i,'decile_score'] + e[1] for i in I1_train]
        max1 =[z[1]-sum((Dtrain.loc[i,"is_recid"]-a[0]*int(Dtrain.loc[i,'age']<25) - c[0]*Dtrain.loc[i,'priors_total_count'] - d[0]*Dtrain.loc[i,'decile_score'] + e[0])**2 for i in I0_train)/len(I0_train)]
        max2 =[z[2]-sum((Dtrain.loc[i,"is_recid"]-a[1]*int(Dtrain.loc[i,'age']<25) - c[1]*Dtrain.loc[i,'priors_total_count'] - d[1]*Dtrain.loc[i,'decile_score'] + e[1])**2 for i in I1_train)/len(I1_train)]
        
        obj_D = z[0] + z[1] + z[2] + 0.05*(e[0]**2 + e[1]**2) #0.05*(e[0]**2 + e[1]**2) 24/02/2023
 
    elif method=='instant-fair':

        # Decision Variables
        a = generate_operators("a", n_vars=2, hermitian=True, commutative=False) # age < 25
        c = generate_operators("c", n_vars=2, hermitian=True, commutative=False) # total number of previous convictions
        d = generate_operators("d", n_vars=2, hermitian=True, commutative=False)
        e = generate_operators("e", n_vars=2, hermitian=True, commutative=False)
        z = generate_operators("z", n_vars=2, hermitian=True, commutative=True)

        # Constraints
        ine1 = [(z[0]+Dtrain.loc[i,"is_recid"] - a[0]*int(Dtrain.loc[i,'age']<25) - c[0]*Dtrain.loc[i,'priors_total_count'] - d[0]*Dtrain.loc[i,'decile_score'] + e[0])/len(I0_train) for i in I0_train]
        ine2 = [(z[0]-Dtrain.loc[i,"is_recid"] + a[0]*int(Dtrain.loc[i,'age']<25) + c[0]*Dtrain.loc[i,'priors_total_count'] + d[0]*Dtrain.loc[i,'decile_score'] + e[0])/len(I0_train) for i in I0_train]
        ine3 = [(z[0]+Dtrain.loc[i,"is_recid"] - a[1]*int(Dtrain.loc[i,'age']<25) - c[1]*Dtrain.loc[i,'priors_total_count'] - d[1]*Dtrain.loc[i,'decile_score'] + e[1])/len(I1_train) for i in I1_train]
        ine4 = [(z[0]-Dtrain.loc[i,"is_recid"] + a[1]*int(Dtrain.loc[i,'age']<25) + c[1]*Dtrain.loc[i,'priors_total_count'] + d[1]*Dtrain.loc[i,'decile_score'] + e[1])/len(I1_train) for i in I1_train]
        max1 = [(z[1]+(Dtrain.loc[i,"is_recid"]-a[0]*int(Dtrain.loc[i,'age']<25) - c[0]*Dtrain.loc[i,'priors_total_count'] - d[0]*Dtrain.loc[i,'decile_score'] + e[0]))/len(I0_train) for i in I0_train]
        max2 = [(z[1]-(Dtrain.loc[i,"is_recid"]-a[1]*int(Dtrain.loc[i,'age']<25) - c[1]*Dtrain.loc[i,'priors_total_count'] - d[1]*Dtrain.loc[i,'decile_score'] + e[1]))/len(I1_train) for i in I1_train]
       
        obj_D = z[0] + 5*z[1] + 0.05*(e[0]**2 + e[1]**2) # 0.05*(e[0]**2 + e[1]**2) 24/02/2023
    
    ine_D=ine1+ine2+ine3+ine4+max1+max2
    sdp_D = SdpRelaxation(variables = flatten([a,c,d,e,z]), verbose = 0)
    sdp_D.get_relaxation(level, objective=obj_D, inequalities=ine_D)
    sdp_D.solve(solver='mosek')
    #sdp_D.solve(solver='sdpa', solverparameters={"executable":"sdpa_gmp","executable": "C:\\Users\\zhouq\\Documents\\sdpa7-windows\\sdpa.exe"})
    #print(sdp_D.primal, sdp_D.dual, sdp_D.status)
        
    return [sdp_D[a[0]],sdp_D[a[1]],sdp_D[c[0]],sdp_D[c[1]],sdp_D[d[0]],sdp_D[d[1]],sdp_D[e[0]],sdp_D[e[1]]]

def normalize_score1(arr):
    # Min-Max normalized scores after deleting outliers.
     #Outliers are set to 0 (if too small) or 1 (if too large) directly.
    outlierPosition=detect_outliers(arr,3)
    arr_clean = np.delete(arr,outlierPosition)
    #arr_clean=arr
    arr_min=np.min(arr_clean)
    arr_max=np.max(arr_clean)

    normalized_arr = np.array([round(float(x - arr_min)/(arr_max - arr_min),1) for x in arr])
    #arr_nor = [int(10*float(x - np.mean(arr)/np.std(arr)) ) for x in arr]
    normalized_arr[normalized_arr>10]=1
    normalized_arr[normalized_arr<0]=0
    return normalized_arr

def detect_outliers(data,threshold):
    # return the location of outliers.
    mean_d = np.mean(data)
    std_d = np.std(data)
    outliers = []
    for i in range(len(data)):
        z_score= (data[i] - mean_d)/std_d 
        if np.abs(z_score) > threshold:
            outliers.append(i)
    return outliers
    
def new_postprocess(dataframe,Itrain,Itest,method):
    a0,a1,c0,c1,d0,d1,e0,e1=training_process(dataframe,Itrain,method)  
    arr=[]
    for i in Itest:
        if dataframe.loc[i,'race']=='Caucasian':
            arr+=[a0*int(dataframe.loc[i,'age']<25) + c0*dataframe.loc[i,'priors_total_count'] + d0*dataframe.loc[i,'decile_score'] + e0]
        elif dataframe.loc[i,'race']!='Caucasian':
            arr+=[a1*int(dataframe.loc[i,'age']<25) + c1*dataframe.loc[i,'priors_total_count'] + d1*dataframe.loc[i,'decile_score'] + e1]
    
    normalized_arr=normalize_score1(np.array(arr))
    return normalized_arr 

def perf_measure(y_actual, y_hat):
# Output: Positive rate, False positive rate; False negative rate; True positive rate
# Positive event is being predicted not to re-offend  0
# Negative event is being predicted to re-offend  1
    TP = 0
    TN = 0
    FP = 0 # False Positive
    FN = 0 # False Negative
    
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==0:
            TP += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==1:
            TN += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FN += 1
    
    #PR =(TP+FP)/len(y_hat) # Positive rate
    NR =(TN+FN)/len(y_hat) # Negative rate
    FPR=FP/(FP+TN+10**(-6)) # False positive rate # + 0.001 to avoid being divided by zero # 干了坏事没被发现
    #TPR=TP/(TP+FN+10**(-6)) # True positive rate # recall/sensitivity # 没干坏事没被冤枉
    FNR=FN/(FN+TP+10**(-6)) # False Omission Rate # 没干坏事却被冤枉
    PPV=TP/(TP+FP+10**(-6)) # positive predictive value (PPV) precision #  成功预测出好人
    NPV=TN/(TN+FN+10**(-6)) # negative predictive value (NPV) # 成功预测出坏人
    inACC=FP+FN # False prediction number
    
    return [NR,FPR,FNR,PPV,NPV,inACC]

def fair_metric(Dtest,I0_test,I1_test,score_name,thresh,base_rate):
    # base_rate override thresh
    if len(base_rate)!=0:
        th0=np.percentile(Dtest[score_name],[base_rate[0]])[0]
        th1=np.percentile(Dtest[score_name],[base_rate[1]])[0]
    else:
        th0=np.percentile(Dtest[score_name],[thresh])[0]
        th1=th0
  
    #print('is I0_test = I1_test: ',len(I0_test)==len(I1_test))
    y_actual_I0=Dtest.loc[I0_test,"is_recid"].tolist()
    y_hat_I0=(Dtest.loc[I0_test,score_name]>=th0).astype(int).tolist()
    #y_compas_I0=(compas_test.loc[I0_test,'compas_decile_score']>=th).astype(int).tolist()

    y_actual_I1=Dtest.loc[I1_test,"is_recid"].tolist()
    y_hat_I1=(Dtest.loc[I1_test,score_name]>=th1).astype(int).tolist()
    #y_compas_I1=(compas_test.loc[I1_test,'compas_decile_score']>=th).astype(int).tolist()
    
    perf_I0=perf_measure(y_actual_I0, y_hat_I0)
    perf_I1=perf_measure(y_actual_I1, y_hat_I1)

    IND=abs(perf_I0[0]-perf_I1[0])
    SP =abs(perf_I0[1]-perf_I1[1]+abs(perf_I1[2]-perf_I0[2]))  # abs(perf_I0[1]-perf_I1[1])+
    #print((perf_I0[2]-perf_I1[2]))
    SF =abs(perf_I0[3]-perf_I1[3]+abs(perf_I0[4]-perf_I1[4])) #abs(perf_I0[3]-perf_I1[3])+
    #print((perf_I0[4]-perf_I1[4]))
    INA=(perf_I0[5]+perf_I1[5])/(len(I0_test)+len(I1_test)) #Dtest.shape[0] # perdiction performance -- inaccuracy
    return [IND,SP,SF,INA]
import sys
sys.path.append("/home/zhouqua1/FairNCPOP") 
sys.path.append("/home/zhouqua1/FairNCPOP/data") 
from functions import*
import numpy as np
import pandas as pd

level=1
filepath = '/home/zhouqua1/FairNCPOP/data/compas-scores-two-years.csv'
nrows=1200

features=['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count','priors_count', 'c_charge_degree']
labels=["is_recid",'decile_score']
performace=['IND','SP','SF','INA','INDrw','SPrw','SFrw','INArw']

unprivileged_groups = [{'race': 1}]  
privileged_groups = [{'race': 0}]

dataframe,base_rate=preprocess(filepath,nrows,features+labels)

## =================
trials=50

methods=["subgroup-fair","instant-fair"] #,"weighted"

metric=pd.DataFrame(columns=performace+['type','thresh','trial'])
for ignore in range(trials):
    Itrain,Itest=split_index(dataframe.index.tolist())
    Dtest=dataframe.loc[Itest]
    I0_test=Dtest[Dtest['race']=='Caucasian'].index
    I1_test=Dtest[Dtest['race']!='Caucasian'].index
    
    # update scores:
    Dtest['COMPAS']=Dtest['decile_score']/10
    for m in methods[:]:
        if m not in ['subgroup-fair',"instant-fair"]:
            pp=aif_postprocess(dataframe,Itrain,Itest,m).flatten()
        else:
            pp=new_postprocess(dataframe,Itrain,Itest,m) #,levels
        i=0
        for j in Itest:
            Dtest.loc[j,m]=pp[i]
            i+=1

    len_rw=min(len(I0_test),len(I1_test))
    I0_test_rw=I0_test[range(len_rw)]
    I1_test_rw=I1_test[range(len_rw)]
    Dtest_rw=Dtest.loc[I0_test_rw.tolist()+I1_test_rw.tolist()]
    
    #update labels, based on thresholds
    #all_thresh = np.linspace(0.2, 0.8, 10)
    #for thresh in all_thresh:  
    thresh=[]
    for m in methods:
        model_perf=fair_metric(Dtest,I0_test,I1_test,m,thresh,base_rate)
        model_perf_rw=fair_metric(Dtest_rw,I0_test_rw,I1_test_rw,m,thresh,base_rate)
        metric=metric.append({'IND':model_perf[0],'SP':model_perf[1],'SF':model_perf[2],'INA':model_perf[3],
                              'INDrw':model_perf_rw[0],'SPrw':model_perf_rw[1],'SFrw':model_perf_rw[2],'INArw':model_perf_rw[3],
                              'type':m,'thresh':0.5,'trial':ignore},ignore_index=True)

metric.to_csv('/home/zhouqua1/FairNCPOP/data/COMPAS4_metric.csv',index=None)

## ====================

"""
trials=5

methods=['COMPAS',"fnr", "fpr", "weighted",'RejectOption','EqOddsPostprocessing'] #

metric=pd.DataFrame(columns=performace+['type','thresh','trial'])

for ignore in tqdm(range(trials)):
    
    Itrain,Itest=split_index(dataframe.index.tolist())
    Dtest=dataframe.loc[Itest]
    I0_test=Dtest[Dtest['race']=='Caucasian'].index
    I1_test=Dtest[Dtest['race']!='Caucasian'].index

    # update scores:
    Dtest['COMPAS']=Dtest['decile_score']/10
    for m in methods[1:]:
        if m not in ['subgroup-fair',"instant-fair"]:
            pp=aif_postprocess(Itrain,Itest,m).flatten()
        else:
            pp=new_postprocess(Itrain,Itest,m)
        i=0
        for j in Itest:
            Dtest.loc[j,m]=pp[i]
            i+=1
            
    I1_test_rw=I1_test[range(len(I0_test))]
    Dtest_rw=Dtest.loc[I0_test.tolist()+I1_test_rw.tolist()]

    # update labels, based on thresholds
    all_thresh = np.linspace(20, 80, 10)
    for thresh in all_thresh:  
        for m in methods:
            model_perf=fair_metric(Dtest,I0_test,I1_test,m,thresh,[]) 
            model_perf_rw=fair_metric(Dtest_rw,I0_test,I1_test_rw,m,thresh,[])
            metric=metric.append({'IND':model_perf[0],'SP':model_perf[1],'SF':model_perf[2],'INA':model_perf[3],
                                  'INDrw':model_perf_rw[0],'SPrw':model_perf_rw[1],'SFrw':model_perf_rw[2],'INArw':model_perf_rw[3],
                                  'type':m,'thresh':round(thresh),'trial':ignore},ignore_index=True)

metric.to_csv('/home/zhouqua1/FairNCPOP/data/AIF3_metric.csv',index=None)

## ===================
trials=50

methods=['COMPAS',"subgroup-fair","instant-fair"] #,"weighted"

metric=pd.DataFrame(columns=performace+['type','thresh','trial'])

for ignore in range(trials):
    
    Itrain,Itest=split_index(dataframe.index.tolist())
    Dtest=dataframe.loc[Itest]
    I0_test=Dtest[Dtest['race']=='Caucasian'].index
    I1_test=Dtest[Dtest['race']!='Caucasian'].index

    # update scores:
    Dtest['COMPAS']=Dtest['decile_score']/10
    for m in methods[1:]:
        if m not in ['subgroup-fair',"instant-fair"]:
            pp=aif_postprocess(dataframe,Itrain,Itest,m).flatten()
        else:
            pp=new_postprocess(dataframe,Itrain,Itest,m)
        i=0
        for j in Itest:
            Dtest.loc[j,m]=pp[i]
            i+=1
            
    I1_test_rw=I1_test[range(len(I0_test))]
    Dtest_rw=Dtest.loc[I0_test.tolist()+I1_test_rw.tolist()]

    # update labels, based on thresholds
    all_thresh = np.linspace(20, 80, 10)
    for thresh in all_thresh:  
        for m in methods:
            model_perf=fair_metric(Dtest,I0_test,I1_test,m,thresh,[]) 
            model_perf_rw=fair_metric(Dtest_rw,I0_test,I1_test_rw,m,thresh,[])
            metric=metric.append({'IND':model_perf[0],'SP':model_perf[1],'SF':model_perf[2],'INA':model_perf[3],
                                  'INDrw':model_perf_rw[0],'SPrw':model_perf_rw[1],'SFrw':model_perf_rw[2],'INArw':model_perf_rw[3],
                                  'type':m,'thresh':round(thresh),'trial':ignore},ignore_index=True)

metric.to_csv('/home/zhouqua1/FairNCPOP/data/COMPAS4_metric_thresh.csv',index=None)
"""

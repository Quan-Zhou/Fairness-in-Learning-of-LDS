#using Pkg
#Pkg.add("CSV")
#Pkg.add(path="https://github.com/wangjie212/TSSOS")

#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --error=/home/zhouqua1/FairNCPOP/err.err 
#SBATCH --output=/home/zhouqua1/FairNCPOP/out.out 
#SBATCH --mem-per-cpu 64G
#SBATCH --time 1-00:00:00 
#SBATCH --partition=amd

using CSV
using Random
using Statistics
using DataFrames

using TSSOS
using DynamicPolynomials

function fairA_timing(Y,d,traj,stp,enp,len)
    T=size(Y,1);
    sumTraj = sum(traj);
    # set nc operators
    @polyvar G Fdash m[1:T+1] q[1:T] z p[1:T] f[1:T]; # all varaibles are assumed to be nc
    var=vcat(z,p,f,G,Fdash,m,q);
    
    # constraints
    ine1 = [f[i] - p[i] - Fdash*m[i+1] for i in 1:T];
    ine2 = [- f[i] + p[i] + Fdash*m[i+1] for i in 1:T];
    ine3 = [m[i+1] - q[i] - G*m[i] for i in 1:T];
    ine4 = [- m[i+1] + q[i] + G*m[i] for i in 1:T];
    max1 = [z-1/traj[1]*sum( 1/len[j]*sum((Y[t,j]-f[t])^2 for t in stp[j]:enp[j]) for j in 1:traj[1])];
    max2 = [z-1/traj[2]*sum( 1/len[j]*sum((Y[t,j]-f[t])^2 for t in stp[j]:enp[j]) for j in traj[1]+1:sumTraj)];
    #max3 = [z-(Y[t,j]-f[t])^2 for j in 1:sumTraj for t in stp[j]:enp[j]];
    
    # objective
    objF= z + 1*sum(p[i]^2 for i in 1:T) + 1*sum(q[i]^2 for i in 1:T);

    # pop
    popA=vcat(objF,ine1,ine2,ine3,ine4,max1,max2);
    
    # solve model fair_A
    return @elapsed tssos_first(popA,var,d,TS="MD",solution=false)

    end

function fairB_timing(Y,d,traj,stp,enp,len)
    T=size(Y,1);
    sumTraj = sum(traj);
    # set nc operators
    @polyvar G Fdash m[1:T+1] q[1:T] z p[1:T] f[1:T]; # all varaibles are assumed to be nc
    var=vcat(z,p,f,G,Fdash,m,q);
    
    # constraints
    ine1 = [f[i] - p[i] - Fdash*m[i+1] for i in 1:T];
    ine2 = [- f[i] + p[i] + Fdash*m[i+1] for i in 1:T];
    ine3 = [m[i+1] - q[i] - G*m[i] for i in 1:T];
    ine4 = [- m[i+1] + q[i] + G*m[i] for i in 1:T];
    #max1 = [z-1/traj[1]*sum( 1/len[j]*sum((Y[t,j]-f[t])^2 for t in stp[j]:enp[j]) for j in 1:traj[1])];
    #max2 = [z-1/traj[2]*sum( 1/len[j]*sum((Y[t,j]-f[t])^2 for t in stp[j]:enp[j]) for j in traj[1]+1:sumTraj)];
    max3 = [z-(Y[t,j]-f[t])^2 for j in 1:sumTraj for t in stp[j]:enp[j]];
    
    # objective
    objF= z + 1*sum(p[i]^2 for i in 1:T) + 1*sum(q[i]^2 for i in 1:T);
    
    # pop
    popB=vcat(objF,ine1,ine2,ine3,ine4,max3);

    # solve model fair_B
    return @elapsed tssos_first(popB,var,d,TS="MD",solution=false)
    #blockcpop_first(popB,var,d,method="chordal") #,method="chordal"

    end

# read observations
Y=CSV.read("/home/zhouqua1/FairNCPOP/data/FairOutput.csv",header=1,DataFrame);
#Y= DataFrame(CSV.File("/home/zhouqua1/FairNCPOP/data/FairOutput.csv"))

# set parameters
d=1
traj = [2,2]

# pre run (because the first time using tssos is very slow)
t=5
stp=vcat(1,rand(1:3,traj[1]-1),1,rand(1:2,traj[2]-1))
enp=vcat(rand(t-2:t,traj[1]-1),t,rand(t-1:t,traj[2]-1),t)
len=[enp[j]-stp[j]+1 for j in eachindex(stp)]
fairA_timing(Y[1:t,:],d,traj,stp,enp,len);
fairA_timing(Y[1:t,:],d,traj,stp,enp,len);

Amean=Float64[]
Astd=Float64[]
Bmean=Float64[]
Bstd=Float64[]
for t in 5:30
    Am=[]
    Bm=[]
    for r in 1:3
        stp=vcat(1,rand(1:3,traj[1]-1),1,rand(1:2,traj[2]-1))
        enp=vcat(rand(t-2:t,traj[1]-1),t,rand(t-1:t,traj[2]-1),t)
        len=[enp[j]-stp[j]+1 for j in eachindex(stp)]
        Ae=fairA_timing(Y[1:t,:],d,traj,stp,enp,len);
        push!(Am,Ae)
        Be=fairB_timing(Y[1:t,:],d,traj,stp,enp,len);
        push!(Bm,Be)
    end
    push!(Amean,mean(copy(Am)))
    push!(Astd,std(copy(Am)))
    push!(Bmean,mean(copy(Bm)))
    push!(Bstd,std(copy(Bm)))
end

A=DataFrame(hcat(Amean,Astd),:auto)#convert(DataFrame,hcat(Amean,Astd))
CSV.write(string("/home/zhouqua1/FairNCPOP/data/tssosAtime0530.csv"),A, writeheader=false)#CSV.write("/home/zhouqua1/FairNCPOP/data/tssosAtime0530.csv",A, writeheader=false)
B=DataFrame(hcat(Bmean,Bstd),:auto)#convert(DataFrame,hcat(Bmean,Bstd))
CSV.write(string("/home/zhouqua1/FairNCPOP/data/tssosBtime0530.csv"),B, writeheader=false)#CSV.write("/home/zhouqua1/FairNCPOP/data/tssosBtime0530.csv",B, writeheader=false)

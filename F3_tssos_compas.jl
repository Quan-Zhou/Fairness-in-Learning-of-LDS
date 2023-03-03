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
using DataFrames
using TSSOS
using DynamicPolynomials

function fairA_timing(Y,d)
    T=size(Y,1);
    # set nc operators
    @polyvar G Fdash m[1:T+1] q[1:T] z p[1:T] f[1:T]; # all varaibles are assumed to be nc
    var=vcat(z,p,f,G,Fdash,m,q);
       
    index11=setdiff(Set(collect(1:T)),Set(findall(ismissing, Y[:,1])));
    index12=setdiff(Set(collect(1:T)),Set(findall(ismissing, Y[:,2])));
    index21=setdiff(Set(collect(1:T)),Set(findall(ismissing, Y[:,3])));
    index22=setdiff(Set(collect(1:T)),Set(findall(ismissing, Y[:,4])));

    # constraints
    ine1 = [f[i] - p[i] - Fdash*m[i+1] for i in 1:T];
    ine2 = [- f[i] + p[i] + Fdash*m[i+1] for i in 1:T];
    ine3 = [m[i+1] - q[i] - G*m[i] for i in 1:T];
    ine4 = [- m[i+1] + q[i] + G*m[i] for i in 1:T];
    max1=[z-( 1/length(index11)*sum((Y[t,1]-f[t])^2 for t in index11)+1/length(index12)*sum((Y[t,2]-f[t])^2 for t in index12) )];
    max2=[z-( 1/length(index21)*sum((Y[t,3]-f[t])^2 for t in index21)+1/length(index22)*sum((Y[t,4]-f[t])^2 for t in index22) )];
    
    # objective
    objF= - z + 1*sum(p[i]^2 for i in 1:T) + 1*sum(q[i]^2 for i in 1:T);

    # pop
    popA=vcat(objF,ine1,ine2,ine3,ine4,max1,max2);
    
    # solve model fair_A
    return @elapsed tssos_first(popA,var,d,TS="MD",solution=false)

    end

function fairB_timing(Y,d)
    T=size(Y,1);
    # set nc operators
    @polyvar G Fdash m[1:T+1] q[1:T] z p[1:T] f[1:T]; # all varaibles are assumed to be nc
    var=vcat(z,p,f,G,Fdash,m,q);
    
    index11=setdiff(Set(collect(1:T)),Set(findall(ismissing, Y[:,1])));
    index12=setdiff(Set(collect(1:T)),Set(findall(ismissing, Y[:,2])));
    index21=setdiff(Set(collect(1:T)),Set(findall(ismissing, Y[:,3])));
    index22=setdiff(Set(collect(1:T)),Set(findall(ismissing, Y[:,4])));

    # constraints
    ine1 = [f[i] - p[i] - Fdash*m[i+1] for i in 1:T];
    ine2 = [- f[i] + p[i] + Fdash*m[i+1] for i in 1:T];
    ine3 = [m[i+1] - q[i] - G*m[i] for i in 1:T];
    ine4 = [- m[i+1] + q[i] + G*m[i] for i in 1:T];
    max31=[z-(Y[t,1]-f[t])^2 for t in index11];
    max32=[z-(Y[t,2]-f[t])^2 for t in index12];
    max33=[z-(Y[t,3]-f[t])^2 for t in index21];
    max34=[z-(Y[t,4]-f[t])^2 for t in index22];

    # objective
    objF= z + 1*sum(p[i]^2 for i in 1:T) + 1*sum(q[i]^2 for i in 1:T);
    
    # pop
    popB=vcat(objF,ine1,ine2,ine3,ine4,max31,max32,max33,max34);

    # solve model fair_B
    return @elapsed tssos_first(popB,var,d,TS="MD",solution=false)

    end

# read observations
Y=CSV.read("/home/zhouqua1/FairNCPOP/data/CompasOutput.csv",header=0,DataFrame);

# set parameters
d=1
traj = [2,2]

# pre run (because the first time using tssos is very slow)
t=5
fairA_timing(Y[1:t,:],d);

Amean=Float64[]
Bmean=Float64[]
for t in 5:21
    Ae=fairA_timing(Y[1:t,:],d);
    Be=fairB_timing(Y[1:t,:],d);
    push!(Amean,copy(Ae))
    push!(Bmean,copy(Be))
end    
A=DataFrame([Amean], [:col1])
B=DataFrame([Bmean], [:col1])
#A=DataFrame(Amean,:auto)#convert(DataFrame,hcat(Amean,Astd))
CSV.write(string("/home/zhouqua1/FairNCPOP/data/tssosAtime0521_compas.csv"),A, writeheader=false)#CSV.write("/home/zhouqua1/FairNCPOP/data/tssosAtime0530.csv",A, writeheader=false)
#B=DataFrame(Bmean,:auto)#convert(DataFrame,hcat(Bmean,Bstd))
CSV.write(string("/home/zhouqua1/FairNCPOP/data/tssosBtime0521_compas.csv"),B, writeheader=false)#CSV.write("/home/zhouqua1/FairNCPOP/data/tssosBtime0530.csv",B, writeheader=false)

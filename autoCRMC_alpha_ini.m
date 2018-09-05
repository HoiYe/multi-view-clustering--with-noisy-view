function [ Us,Ustar,alpha,objs ] = autoCRMC_alpha_ini( Ks,k,c,stopGap,maxIter,alpha_ini)
% autoCRMC_kernel Summary of this function goes here
% using kernel kmeans to find the underlying Us and Uavg without manually
% setting alpha
%   Detailed explanation goes here
% input:
% Ks: 3dim matrix that stores the kernel matrix of each view
% k: the number of classes
% stopGap: stop when obj difference less than this
% maxIter: stop when iterNum equals this
% alpha_ini: initialization of alpha
% output:
% Us: the cluster embeddings of different views
% Ustar: the centroid embedding
% alpha: the weights for correlations between views


opts.disp = 0;
[n,~,vNum]=size(Ks);
Us=zeros(n,k,vNum);
% initial alpha
alpha=alpha_ini;



%% initial Ui
for vI=1:vNum
    [U E] = eigs(Ks(:,:,vI),k,'LA',opts);
    Us(:,:,vI)=U;
end
%% initial Ustar
[Ustar]=updateUstar(Us,k);
%% get the fist obj
iterNum=1;
obj0=calObj(Us,Ustar,alpha,c,Ks);
objs(iterNum)=obj0;
%% update alpha, Us,Ustar iterativly untill objDiff<stopGap
objDiff=inf;

while iterNum<maxIter&&objDiff>stopGap
    iterNum=iterNum+1;
    %% update alpha
    [ alpha ] = updateAlpha( Us,Ks);
    %% update Us
    [Us]=updateUs(Ks,Ustar,alpha,c,k);
    %% update Ustar
    [Ustar]=updateUstar(Us,k);
    %% cal obj
    objs(iterNum)=calObj(Us,Ustar,alpha,c,Ks);
    objDiff=objs(iterNum)-objs(iterNum-1);
    if objDiff<0
        if abs(objDiff)>stopGap
            objDiff
        error('obj decrease')
        end
    end
end
end
function [alpha]=updateAlpha(Us,Ks)

p=size(Ks,3);
d=zeros(p,1);
alpha=zeros(p,1);
for j=1:p
    Uj=Us(:,:,j);
    d(j)=trace(Uj'*Ks(:,:,j)*Uj);
end
alpha=d/sqrt(d'*d);

end
function [obj] =calObj(Us,Ustar,alpha,c,Ks)
vNum=size(Us,3);
obj=0;
for vI=1:vNum
    U=Us(:,:,vI);
    obj=obj+alpha(vI)*trace(U'*Ks(:,:,vI)*U)+c*trace((U'*Ustar)*(Ustar'*U));
end

end


function [Ustar]=updateUstar(Us,k)
[n,~,vNum]=size(Us);
Kavg=zeros(n,n);
for vI=1:vNum
    U=Us(:,:,vI);
    Kavg=Kavg+(U*U');
end
opts.disp = 0;
[Ustar, ~]=eigs(Kavg,k,'LA',opts);
end

function [Us]=updateUs(Ks,Ustar,alpha,c,k)
opts.disp=0;
[n,~,vNum]=size(Ks);
Ku=zeros(n,n);
Us=zeros(n,k,vNum);
for vI=1:vNum
    Ku=alpha(vI)*Ks(:,:,vI)+c*(Ustar*Ustar');
    [Us(:,:,vI) E]=eigs(Ku,k,'LA',opts);
end
end



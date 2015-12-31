function [like] = getMixGaussLike(datum,mixGaussEst)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here



%nData=400
%initialize log likelihoods

like=0.0;
D=mixGaussEst.d;
for k=1:mixGaussEst.k
    SigmaDet=det(mixGaussEst.cov(:,:,k));
    meanMat=mixGaussEst.mean(:,k); 
    like=like+...
        mixGaussEst.weight(1,k)*( 1/(((2*pi)^(D/2))*(SigmaDet^(1/2))) )*...
        (exp( -0.5*(datum-meanMat)'*(mixGaussEst.cov(:,:,k)^-1)*(datum-meanMat) ));
end


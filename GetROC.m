function [TPR FPR P N TP FN TN FP] = GetROC( img,groundTruth )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
TP=zeros(256,1);
FN=zeros(256,1);
TN=zeros(256,1);
FP=zeros(256,1);

for T=0:255
    
    ThresholdedImage=thresholdImage(img,T);
    %ThresholdedImage=changeRangeOfImage(ThresholdedImage);
    
    for x=1:size(img,1)
        for y=1:size(img,2)
            if( groundTruth(x,y)==1 && ThresholdedImage(x,y)==1 )
                TP(T+1)=TP(T+1)+1;
            elseif( groundTruth(x,y)==1 && ThresholdedImage(x,y)==0 )
                FN(T+1)=FN(T+1)+1;
            elseif(groundTruth(x,y)==0 && ThresholdedImage(x,y)==0)
                TN(T+1)=TN(T+1)+1;
            elseif(groundTruth(x,y)==0 && ThresholdedImage(x,y)==1)
                FP(T+1)=FP(T+1)+1;
            end     
        end
    end
    
end


P=TP+FN;
N=FP+TN;

TPR=TP./P;
FPR=FP./N;


end


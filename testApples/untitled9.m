im1=imread('Bbr98ad4z0A-ctgXo3gdwu8-original.jpg');
gt=imread('Bbr98ad4z0A-ctgXo3gdwu8-original.png');


[TPR FPR P N TP FN TN FP] = GetROC( posteriorApple,GT_TestImagesCells{imgIndex} );
        fig1=figure;
        figure(fig1);
        plot(FPR,TPR,'b*');
        str = sprintf('ROC of Image: %d-Part 1',imgIndex);
        title(str);
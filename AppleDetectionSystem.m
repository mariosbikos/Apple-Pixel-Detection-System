% AppleDetectionSystem.m
clear all;
close all;


%%
%Check if we are in the correct path for loading images
if( ~exist('apples', 'dir') || ~exist('testApples', 'dir') || ~exist('new_apples', 'dir') )
    display('Please change current directory to the parent folder of apples/,testApples/ and new_apples/');
end

%Load the individual path for each image used for training (3 in total)
Iapples = cell(3,1);
Iapples{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.jpg';
Iapples{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.jpg';
Iapples{3} = 'apples/bobbing-for-apples.jpg';
%Load the individual path for each Ground Truth Image used for training (3 in total)
IapplesMasks = cell(3,1);
IapplesMasks{1} = 'apples/Apples_by_kightp_Pat_Knight_flickr.png';
IapplesMasks{2} = 'apples/ApplesAndPears_by_srqpix_ClydeRobinson.png';
IapplesMasks{3} = 'apples/bobbing-for-apples.png';


%Load the individual path for each image of the 1st Testing Set of images
%(3 in total)
ITestapples = cell(3,1);
ITestapples{1} = 'testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.jpg';
ITestapples{2} = 'testApples/audioworm-QKUJj2wmxuI-original.jpg';
ITestapples{3} = 'testApples/Apples_by_MSR_MikeRyan_flickr.jpg';

%Load the path for the Ground Truth image of the 1st image 
%of the 1st Testing set of images(given)
ITestapplesMasks = cell(1,1);
ITestapplesMasks{1} = 'testApples/Bbr98ad4z0A-ctgXo3gdwu8-original.png';

%Load the paths for for each image of the 2nd Testing Set of images
%(found on the web)
ITestNewapples = cell(2,1);
ITestNewapples{1} = 'new_apples/15756-a-plate-of-red-rome-beauty-apples-pv.jpg';
ITestNewapples{2} = 'new_apples/4682627_3dba83c8.jpg';

%Load the path for the Ground Truth image of the images
%of the 2nd Testing set of images(created by me)
ITestNewapplesMasks = cell(2,1);
ITestNewapplesMasks{1} = 'new_apples/15756-a-plate-of-red-rome-beauty-apples-pv.png';
ITestNewapplesMasks{2} = 'new_apples/4682627_3dba83c8.png';

%%
%Create the training vector sets RGBApple,RGBNonApple which
%include the RGB values for every single pixel of the 3 images
%depending on whether they belong to apple pixels or not(We find this using
%GT images
numDimensions=3;
RGBApple=zeros(numDimensions,1);
RGBNonApple=zeros(numDimensions,1);
counterOfApplePixels=1;
counterOfNonApplePixels=1;

%Load the training images and transform them to data types that we can use
for iImage=1:3
    curI = double(imread(  Iapples{iImage}   )) / 255;
    % curI is now a double-precision 3D matrix of size (width x height x 3).
    % Each of the 3 color channels is now in the range [0.0, 1.0].
    
    %Load GT
    curImask = imread(  IapplesMasks{iImage}   );
    % These mask-images are often 3-channel, and contain grayscale values. We
    % would prefer 1-channel and just binary:
    curImask = curImask(:,:,2) > 128;  % Picked green-channel arbitrarily.
    
    %Fill the training arrays RGBApple,RGBNonApple
    %taking into account the value of each pixel 
    %in the corresponding mask image
    for x=1:size(curImask,1)
        for y=1:size(curImask,2)
            if(curImask(x,y)==1)
                RGBApple(1:3,counterOfApplePixels)=curI(x,y,1:3);
                counterOfApplePixels=counterOfApplePixels+1;
            elseif(curImask(x,y)==0)
                RGBNonApple(1:3,counterOfNonApplePixels)=curI(x,y,1:3);
                counterOfNonApplePixels=counterOfNonApplePixels+1;
            end
        end
    end
end

%%
%Set number of gaussians for each MoG
k=2;
%Now we have the training data vectors for apple and non-apple pixels
%----------------------------------------------------------------------%
%TRAIN THE MODEL TO ESTIMATE THE WEIGHTS,MEANS,COVARIANCES FOR EACH MOG
ApplemixGaussEst = TrainModel( RGBApple,k );
NonApplemixGaussEst =TrainModel( RGBNonApple,k );

%%
%Now inside the structs ApplemixGaussEst and NonApplemixGaussEst we have
%the weights,means and covariances we need in order to calculate the
%likelihood for test images

%now run through the pixels in the test image and classify them as being skin or
%non skin - we will fill in the posterior
%let's define priors for whether the pixel is skin or non skin
%This can be done counting the percentage of white pixels and black pixels 
%of the ground truth of the images used for training
fprintf('Training Apple MoG Model...');
%priorApple = (counterOfApplePixels-1)/((counterOfApplePixels-1)+(counterOfNonApplePixels-1));
priorApple =0.3;
fprintf('Training Non-Apple MoG Model...');
%priorNonApple = (counterOfNonApplePixels-1)/((counterOfApplePixels-1)+(counterOfNonApplePixels-1));
priorNonApple =0.7;
%%
%Create a figure for each of the images of the 1st Set of Test Images
%In order to render in an array form
Test_1_Images_figuresCells=cell(3,1);
for i=1:3
    Test_1_Images_figuresCells{i}=figure;
end
%Load the 1st set of Test Images and the Ground truth for one of them
TestImagesCells=cell(3,1);
GT_TestImagesCells=cell(1,1);
for testImageIndex=1:3
    curI = double(imread(  ITestapples{testImageIndex}   )) / 255;
    TestImagesCells{testImageIndex}=curI;
    figure(Test_1_Images_figuresCells{testImageIndex});
    subplot(1,3,1);
    imshow(TestImagesCells{testImageIndex});
    if testImageIndex==1
        GT_TestImagesCells{testImageIndex}=imread(ITestapplesMasks{testImageIndex});
        GT_TestImagesCells{testImageIndex}=GT_TestImagesCells{testImageIndex}(:,:,2) > 128;
        subplot(1,3,2);
        imshow(GT_TestImagesCells{testImageIndex});
    end
end


%%
Test_2_Images_figuresCells=cell(2,1);
for i=1:2
    Test_2_Images_figuresCells{i}=figure;
end
%LOAD NEW TEST IMAGES(FOUND ONLINE)
%Load the new test images to a 2x1 cell array
NewTestImagesCells=cell(2,1);
GT_NewTestImagesCells=cell(2,1);
for testImageIndex=1:2
    %Image load
    curI = double(imread(  ITestNewapples{testImageIndex}   )) / 255;
    NewTestImagesCells{testImageIndex}=curI;
    figure(Test_2_Images_figuresCells{testImageIndex});
    subplot(1,3,1);
    imshow(NewTestImagesCells{testImageIndex});
    %GT Load
    GT_NewTestImagesCells{testImageIndex}=imread(ITestNewapplesMasks{testImageIndex});
    GT_NewTestImagesCells{testImageIndex}=GT_NewTestImagesCells{testImageIndex}(:,:,2) > 128;
    subplot(1,3,2);
    imshow(GT_NewTestImagesCells{testImageIndex});
end

%%
%For each image of the 2nd set of Test Images
%-->Calculate the posterior probability of each pixel being apple
%   and plot the result image
for imgIndex=1:3
    [imY imX imZ] = size(TestImagesCells{imgIndex});
    %imY are the rows of the image
    posteriorApple = zeros(imY,imX);
    for cY = 1:imY
        fprintf('Processing Row %d\n',cY);
        for cX = 1:imX
            %extract this pixel data --> we get a 3x1 values of pixel
            thisPixelData = squeeze(double(TestImagesCells{imgIndex}(cY,cX,:)));
            %calculate likelihood of this data given skin model
            likeApple=getMixGaussLike(thisPixelData,ApplemixGaussEst);
            %calculate likelihood of this data given non skin model
            likeNonApple = getMixGaussLike(thisPixelData,NonApplemixGaussEst);
            %Calculate posterior probability from likelihoods and
            %priors using BAYES rule.
            posteriorApple(cY,cX) = (likeApple * priorApple)/...
                (likeApple * priorApple+ priorNonApple * likeNonApple);
        end;
    end;
    %draw skin posterior
    clims = [0, 1];
    figure(Test_1_Images_figuresCells{imgIndex});
    subplot(1,3,3);
    imagesc(posteriorApple, clims); colormap(gray); axis off; axis image;
    drawnow;
    str = sprintf('SecondSetPosteriorImage_of_%d',imgIndex);
    print(Test_1_Images_figuresCells{imgIndex},str,'-dpng');
    %posteriorApple has values from 0-1 which show probability of pixel
    %being apple pixel
    %If we threshold the posteriorApple image using T, then we can get 0 or 1.
    
    %If we have the testImage 1, then we can also calculate the ROC Curve
    %since we are given the Ground Truth of the image
    %Therefore, we need to calculate the thresholded image for thresholds
    %from 0-255
    if imgIndex==1
        [TPR FPR P N TP FN TN FP] = GetROC( posteriorApple,GT_TestImagesCells{imgIndex} );
        fig1=figure;
        figure(fig1);
        plot(FPR,TPR,'b');
        str = sprintf('ROC of Image: %d-Part 1',imgIndex);
        title(str);
        drawnow;
        print(fig1,str,'-dpng');
        %Calculate the area below the ROC curve to get a quantitative result
        AreaOfROC=trapz(sort(FPR),sort(TPR));
        fprintf('Area of ROC Curve for the Test Image of Second Set= %1.4f',AreaOfROC);
    end
end

fig_rocs=cell(2,1);
fig_rocs{1}=figure;
fig_rocs{2}=figure;
%%
%For each of the images of the 3rd set of test images
%calculate the posterior probability images and ROC curves
for imgIndex=1:2
    [imY imX imZ] = size(NewTestImagesCells{imgIndex});
    %imY are the rows of the image
    posteriorApple = zeros(imY,imX);
    for cY = 1:imY
        fprintf('Processing Row %d\n',cY);
        for cX = 1:imX
            %extract this pixel data --> we get a 3x1 values of pixel
            thisPixelData = squeeze(double(NewTestImagesCells{imgIndex}(cY,cX,:)));
            %calculate likelihood of this data given skin model
            likeApple=getMixGaussLike(thisPixelData,ApplemixGaussEst);
            %calculate likelihood of this data given non skin model
            likeNonApple = getMixGaussLike(thisPixelData,NonApplemixGaussEst);
            %Calculate posterior probability from likelihoods and
            %priors using BAYES rule.
            posteriorApple(cY,cX) = (likeApple * priorApple)/...
                (likeApple *priorApple+ priorNonApple *likeNonApple);
        end;
    end;
    %draw skin posterior
    clims = [0, 1];
    figure(Test_2_Images_figuresCells{imgIndex});
    subplot(1,3,3);
    imagesc(posteriorApple, clims); colormap(gray); axis off; axis image;
    drawnow;
    str = sprintf('ThirdSetPosteriorImage_of_%d',imgIndex);
    print(Test_2_Images_figuresCells{imgIndex},str,'-dpng');
    %posteriorApple has values from 0-1 which show probability of being apple
    %If we threshold the posteriorApple image using T, then we can get 0 or 1.
    
    %calculate the ROC Curve
    %since we are given the Ground Truth of the image
    [TPR FPR P N TP FN TN FP] = GetROC( posteriorApple,GT_NewTestImagesCells{imgIndex} );
    figure(fig_rocs{imgIndex});
    plot(FPR,TPR,'b');
    str = sprintf('ROC of Image: %d-Part 2',imgIndex);
    title(str);
    drawnow;
    print(fig_rocs{imgIndex},str,'-dpng');
    AreaOfROC=trapz(sort(FPR),sort(TPR));
    fprintf('Area of ROC Curve for New Test Image: %d of Third Set= %1.4f',imgIndex,AreaOfROC)
    
end
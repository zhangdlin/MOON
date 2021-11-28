
%%
% The source code of "MOON: Multi-Hash Codes Joint Learning for Cross-Media
% Retrieval'
%If you have any questions, you can contact me at dlinzzhang@gmail.com.
%If you use our code, please cite our article.
function Main_demo()
clc
clear all;

%% load dataset
load('MIRFlickr25k.mat');
XTrain = I_tr; YTrain = T_tr; LTrain = L_tr;
XTest = I_te; YTest = T_te; LTest = L_te;

%% initialization
fprintf('initializing...\n')
param.lambdaX = 0.5;
param.alpha = 500; % 500
param.Xmu = 1000;
param.gamma = 5; %5
param.iter = 20;

%% tune
param.Xbeide = 1e-6; %1E-x

nbits = 12;
nbits1 = 24;
nbits2 = 36;
nbits3 = 48;


run = 1;

%% centralization
fprintf('centralizing data...\n');
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));

%% kernelization
    param.nXanchors = 1000; param.nYanchors = 1000;
    if 1
        anchor_idx = randsample(size(XTrain,1), param.nXanchors);
        XAnchors = XTrain(anchor_idx,:);
        anchor_idx = randsample(size(YTrain,1), param.nYanchors);
        YAnchors = YTrain(anchor_idx,:);
    else
        [~, XAnchors] = litekmeans(XTrain, param.nXanchors, 'MaxIter', 30);
        [~, YAnchors] = litekmeans(YTrain, param.nYanchors, 'MaxIter', 30);
    end
    
    [XKTrain,XKTest]=Kernel_Feature(XTrain,XTest,XAnchors);
    [YKTrain,YKTest]=Kernel_Feature(YTrain,YTest,YAnchors);




%% evaluation
for i=1:run
    
    %% SCRATCH
    param.nbits=nbits;
    param.nbits1=nbits1;
    param.nbits2=nbits2;
    param.nbits3=nbits3;


    eva_info =evaluate(XKTrain,YKTrain,XKTest,YKTest,LTest,LTrain,param);
    
    % train time
    trainT = eva_info.trainT;
    
    % MAP
    map(i,1) = eva_info.Image_to_Text_MAP01;
    map(i,2)=  eva_info.Text_to_Image_MAP01;
    
    map(i,3) = eva_info.Image_to_Text_MAP02;
    map(i,4)=  eva_info.Text_to_Image_MAP02;
    
    map(i,5) = eva_info.Image_to_Text_MAP03;
    map(i,6)=  eva_info.Text_to_Image_MAP03;
    
    map(i,7) = eva_info.Image_to_Text_MAP04;
    map(i,8)=  eva_info.Text_to_Image_MAP04;
end
    fprintf('MMM %d bits --  Image_to_Text_MAP: %f ; Text_to_Image_MAP: %f ; train time: %f\n\n',nbits,mean(map( : , 1)),mean(map( : , 2)),trainT);
    fprintf('MMM %d bits --  Image_to_Text_MAP: %f ; Text_to_Image_MAP: %f ; train time: %f\n\n',nbits1,mean(map( : , 3)),mean(map( : , 4)),trainT);
    fprintf('MMM %d bits --  Image_to_Text_MAP: %f ; Text_to_Image_MAP: %f ; train time: %f\n\n',nbits2,mean(map( : , 5)),mean(map( : , 6)),trainT);
    fprintf('MMM %d bits --  Image_to_Text_MAP: %f ; Text_to_Image_MAP: %f ; train time: %f\n\n',nbits3,mean(map( : , 7)),mean(map( : , 8)),trainT);


end

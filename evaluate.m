function evaluation_info=evaluate(XTrain,YTrain,XTest,YTest,LTest,LTrain,param)
    tic;
    
    [Wx, Wy, R, B, Wx1, Wy1, R1, B1, Wx2, Wy2, R2, B2, Wx3, Wy3, R3, B3] = train(XTrain, YTrain, param, LTrain);
    
    fprintf('evaluating...\n');
    
    %% Training Time
    traintime=toc;
    evaluation_info.trainT=traintime;

    %% bits one
   %% image as query to retrieve text database
    BxTest = compactbit(XTest*Wx'*R' >= 0);
    ByTrain = compactbit(B' >= 0);
    hri2t = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, BxTest, ByTrain);
    evaluation_info.Image_to_Text_MAP01 = hri2t;


    %% text as query to retrieve image database
    ByTest = compactbit(YTest*Wy'*R' >= 0);
    BxTrain = compactbit(B' >= 0);    
    hrt2i = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, ByTest, BxTrain);
    evaluation_info.Text_to_Image_MAP01 = hrt2i;

 
   %% bits two
   %% image as query to retrieve text database
    BxTest = compactbit(XTest*Wx1'*R1' >= 0);
    ByTrain = compactbit(B1' >= 0);    
    hri2t = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, BxTest, ByTrain);
    evaluation_info.Image_to_Text_MAP02 = hri2t;

    %% text as query to retrieve image database
    ByTest = compactbit(YTest*Wy1'*R1' >= 0);
    BxTrain = compactbit(B1' >= 0);
    hrt2i = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, ByTest, BxTrain);
    evaluation_info.Text_to_Image_MAP02 = hrt2i;

   %% bits three 
   
   %% image as query to retrieve text database
    BxTest = compactbit(XTest*Wx2'*R2' >= 0);
    ByTrain = compactbit(B2' >= 0);
    hri2t = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, BxTest, ByTrain);
    evaluation_info.Image_to_Text_MAP03 = hri2t;


    %% text as query to retrieve image database
    ByTest = compactbit(YTest*Wy2'*R2' >= 0);
    BxTrain = compactbit(B2' >= 0);
    hrt2i = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, ByTest, BxTrain);
    evaluation_info.Text_to_Image_MAP03 = hrt2i;
      
      
    
     %%  case-4
   
   %% image as query to retrieve text database
    BxTest = compactbit(XTest*Wx3'*R3' >= 0);
    ByTrain = compactbit(B3' >= 0);
    hri2t = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, BxTest, ByTrain);
    evaluation_info.Image_to_Text_MAP04 = hri2t;

      
    %% text as query to retrieve image database
      ByTest = compactbit(YTest*Wy3'*R3' >= 0);
      BxTrain = compactbit(B3' >= 0);
  
      hrt2i = calcMapTopkMapTopkPreTopkRecLabel(LTest, LTrain, ByTest, BxTrain);
      evaluation_info.Text_to_Image_MAP04 = hrt2i;
   
  

   
    
   
end
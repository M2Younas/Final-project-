clc;
clear all;
close all;
matlabroot='E:\septemper2019\Desktop\2019new\andrafundus\datas'


DatasetPath = fullfile(matlabroot);

Data = imageDatastore(DatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
     

[trainData] = splitEachLabel(Data,0.8,'randomize');

     
CountLabel = Data.countEachLabel;

%% Define the Network Layers
      layers1=[imageInputLayer([256 256 3])
            convolution2dLayer(11,128,'Padding',0,'stride',4)
            reluLayer
            crossChannelNormalizationLayer(5)
            maxPooling2dLayer(3,'Stride',2,'Padding',0)
            convolution2dLayer(5,512,'Padding',2,'stride',1)
            reluLayer
            crossChannelNormalizationLayer(5)
            maxPooling2dLayer(3,'Stride',2,'Padding',0)
            convolution2dLayer(3,384,'Padding',1,'stride',1)
            reluLayer
            convolution2dLayer(3,256,'Padding',1,'stride',1)
            reluLayer
            convolution2dLayer(3,256,'Padding',1,'stride',1)
            reluLayer
            maxPooling2dLayer(3,'Stride',2,'Padding',0)
            fullyConnectedLayer(5000)
            reluLayer
            dropoutLayer
            fullyConnectedLayer(1000)
            reluLayer
            dropoutLayer
            fullyConnectedLayer(5)
            softmaxLayer
            classificationLayer];
      
      
      
options = trainingOptions('sgdm','MaxEpochs',100, ...
	'InitialLearnRate',0.0001);  

convnet = trainNetwork(trainData,layers1,options);

    
I = imread('test1.jpg');
 
class = classify(convnet,I)

msgbox(char(class))

 



%% use pca to get the eigenvectors and eigenvalues
%% and plot the data points of the first 2 components
%% running environment£º matlab 2016b, whose library contains pca() function

clc
close all
addpath(genpath(pwd));
load('DataA_filled.mat');% get DataB.mat to fea
load('gnd.mat');% get DataB.mat to fea
feat=zeros(12,7);
[G, SCORE, LATENT, TSQUARED] = pca(zscore(feafilled)); % LATENT is the eigenvalues
for(i=1:12)
%g = graycomatrix(G(:,i));
Mean = mean2(G(:,i));
Standard_Deviation = std2(G(:,i));
%Entropy = entropy(G(:,i));
RMS = mean2(rms(G(:,i)));
Variance = mean2(var(double(G(:,i))));
b=G(:,i);
a = sum(double(b(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(b(:)));
Skewness = skewness(double(b(:)));

  
feat(i,:) = [ Mean, Standard_Deviation, RMS, Variance, Smoothness, Kurtosis, Skewness]

end
figure;
gscatter(SCORE(:,9),SCORE(:,12),gnd);
xlabel('X'),ylabel('Y');
title('Classes reprensentation based on the 9th and 12th principal components ');
figure
gscatter(SCORE(:,3),SCORE(:,4),gnd);
xlabel('X'),ylabel('Y');
title('Classes reprensentation based on the 3rd and 4th principal components ');







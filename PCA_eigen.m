%% running environment£º matlab 2016b, whose library contains pca() function
function Q2_1()
addpath(genpath(pwd));
load('DataA_filled.mat');% get DataB.mat to fea
%% use pca to get the eigenvectors and eigenvalues
[eigenvectors, ~, eigenvalues, ~] = pca(zscore(feafilled)); % LATENT is the eigenvalues
save('eigen.mat','eigenvectors','eigenvalues');
end
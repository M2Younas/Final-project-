

%% Ant Colony System
clc, clear, close 
% Benchmark data set 
load ionosphere.mat; 

% Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho,'Stratify',false);

% Parameter setting
N        = 10; 
max_Iter = 100; 
tau      = 1; 
eta      = 1; 
alpha    = 1; 
beta     = 1; 
rho      = 0.2; 
phi      = 0.5; 
Nf       = 15;       % Set number of selected features
% Ant Colony System
[sFeat,Nf,Sf,curve] = jACO(feat,label,N,max_Iter,tau,eta,alpha,beta,rho,phi,Nf,HO);

% Plot convergence curve
plot(1:max_Iter,curve); 
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ACS'); grid on;







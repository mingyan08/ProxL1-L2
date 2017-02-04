clear; close all
clc

%% parameter settings
M = 64; N = 256;    % matrix dimension M-by-N
K = 10;             % sparsity

matrixtype = 1; 
% 1 for Gaussian
% 2 for DCT


%% construct sensing matrix
switch matrixtype 
    case 1
        A   = randn(M,N); % Gaussian matrix
        A   = A / norm(A);    
    case 2    
        A   = dctmtx(N); % dct matrix
        idx = randperm(N-1);
        A   = A([1 idx(1:M-1)+1],:); % randomly select m rows but always include row 1
        A   = A / norm(A);   
end


%% construct sparse ground-truth 
x_ref = zeros(N,1); % true vector
xs = randn(K,1);
x_ref(randsample(N,K)) = xs;


%% Given lambda, construct b, so that x is a stationary point
lambda = 1e-2;

x = x_ref;
[b,y,w,output] = construct_test4L12(A,x,lambda);

% check optimality
norm(lambda * (w - x / norm(x)) + A' * (A * x - b))
max([max(w)-1,min(w) + 1,norm(w(x > 0) - 1),norm(w(x < 0) + 1)])



%% compare L1-L2 solvers
pm.lambda = lambda;
pm.delta = pm.lambda*100;
pm.xg = x; 
pmFB = pm; pmFB.delta =  1;


% initialization
[xDCA,outputDCA] = CS_L1L2_uncon_DCA(A,b,pm);
[xADMM,outputADMM] = CS_L1L2_uncon_ADMM(A,b,pm);
[xFB,outputFB] = CS_L1L2_uncon_FB(A,b,pmFB);


figure
plot(log10(outputDCA.err), 'r', 'LineWidth',2)
hold on
plot(log10(outputFB.err),'k-.', 'LineWidth',2);
hold on
plot(log10(outputADMM.err),'b--', 'LineWidth',2)
LEG = legend('DCA',  'FBS','ADMM', 'location', 'NorthEast');

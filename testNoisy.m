clear; close all
clc

%% parameter settings
M = 250; N = 512;   % matrix dimension M-by-N
K = 130;            % sparsity


for trial = 1:100
        trial
   
        A   = randn(M,N); 
        A   = orth(A')';    % normalize each column to be zero mean and unit norm
        
        %% construct sparse ground-truth 
        x_ref       = zeros(N,1); % true vector
        xs          = randn(K,1);
        idx         = randperm(N);
        supp        = idx(1:K);
        x_ref(supp) = xs;
        As          = A(:,supp);

        sigma       = 0.1;
        b           = A * x_ref + sigma * randn(M,1); 
        
        MSEoracle(trial) = sigma^2 * trace(inv(As' * As));


        %% parameters
        pm.lambda = 0.08;
        pm.delta = normest(A*A',1e-2)*sqrt(2);
        pm.xg = x_ref; 
        pmL1 = pm; 
        pmL1.maxit = 2*N;
        
        
        %% initialization with inaccurate L1 solution
        x1      = CS_L1_uncon_ADMM(A,b,pmL1); 
        pm.x0   = x1;   
        
        
        %% L1-L2 implementations
        xDCA            = CS_L1L2_uncon_DCA(A,b,pm);
        xADMM           = CS_L1L2_uncon_ADMM(A,b,pm);
        xADMMweighted   = CS_L1L2_uncon_ADMMweighted(A,b,pm);

        pmFB = pm; pmFB.delta = 1;
        [xFB,outputFB] = CS_L1L2_uncon_FBweighted(A,b,pmFB);

        
        %% compute MSE
        xall = [x1 xDCA, xADMM, xADMMweighted,xFB];
        for k = 1:size(xall,2)  
            xx = xall(:,k);
            MSE(trial, k) =norm(xx-x_ref);
        end

end

[mean(MSEoracle), mean(MSE,1)]
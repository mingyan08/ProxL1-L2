clc, close all; clear
M = 100; N = 1500;      % matrix dimension M-by-N
K = 25;                 % sparsity
F = 20;                 % larger for higher coherence


%% parameters
pm.lambda = 1e-7; pm.maxit = 5*N;
pmL1 = pm; pmL1.maxit = 2*N;

% how to dynamically update alpha
if F>10
    pm.alpha_update = 2;
else
    pm.alpha_update = 1;
end


%% highly coherent matrix
A = zeros(M,N);
r = rand(M,1);
l = 1:N;
for k = 1:M
        A(k,:) = sqrt(2/M) * cos(2 * pi * r(k) * (l-1) / F);
end
        
A = A/norm(A);
        
%% sparse vector with minimum separation
supp        = randsample_separated(N,K,2*F);
x_ref       = zeros(N,1);
xs          = randn(K,1);
x_ref(supp) = xs;
b           = A * x_ref;

%% initialize by an inaccurate L1 solution
[x1,output] = CS_L1_uncon_ADMM(A,b,pmL1);
pm.x0       = x1;

xDCA            = CS_L1L2_uncon_DCA(A,b,pm);
xADMM           = CS_L1L2_uncon_ADMM(A,b,pm);
xADMMweighted   = CS_L1L2_uncon_ADMMweighted(A,b,pm);
        
%% exact L1 solution as baseline
[x1,output] = CS_L1_uncon_ADMM(A,b,pm);

log10([norm(x1-x_ref), norm(xDCA-x_ref), norm(xADMM-x_ref), norm(xADMMweighted-x_ref)]/norm(x_ref))
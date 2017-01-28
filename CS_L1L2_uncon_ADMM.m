function [x, output] = CS_L1L2_uncon_ADMM(A, b, pm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%         min_x .5||Ax-b||^2 + lambda(|x|_1- alpha |x|_2)             %%%  
%%%                                                                     %%%
%%% Input: dictionary A, data b, parameters set pm                      %%%
%%%        pm.lambda: regularization paramter                           %%%
%%%        pm.delta: penalty parameter for ADMM                         %%%
%%%        pm.maxit: max iterations                                     %%%
%%%        pm.reltol: rel tolerance for ADMM: default value: 1e-6       %%%
%%%        pm.alpha: alpha in the regularization                        %%%
%%% Output: computed coefficients x                                     %%%
%%%        output.relerr: relative error of yold and y                  %%%
%%%        output.obj: objective function of x_n:                       %%%
%%%        obj(x) = lambda(|x|_1 - alpha |x|_2)+0.5|Ax-b|^2             %%%
%%%        output.res: residual of x_n: norm(Ax-b)/norm(b)              %%%
%%%        output.err: error to the ground-truth: norm(x-xg)/norm(xg)   %%%
%%%        output.time: computational time                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[M,N]       = size(A); 
start_time  = tic; 


%% parameters
if isfield(pm,'lambda'); 
    lambda = pm.lambda; 
else
    lambda = 1e-5;  % default value
end
% parameter for ADMM
if isfield(pm,'delta'); 
    delta = pm.delta; 
else
    delta = 100 * lambda;
end
% maximum number of iterations
if isfield(pm,'maxit'); 
    maxit = pm.maxit; 
else 
    maxit = 5*N; % default value
end
% initial guess
if isfield(pm,'x0'); 
    x0 = pm.x0; 
else 
    x0 = zeros(N,1); % initial guess
end
if isfield(pm,'xg'); 
    xg = pm.xg; 
else 
    xg = x0;
end
if isfield(pm,'reltol'); 
    reltol = pm.reltol; 
else 
    reltol  = 1e-6; 
end
if isfield(pm,'alpha'); 
    alpha = pm.alpha;
else 
    alpha = 1;
end

%% pre-computing/initialize
AAt     = A*A';
L       = chol(speye(M) + 1/delta*AAt, 'lower');
L       = sparse(L);
U       = sparse(L');   

x       = zeros(N,1);
Atb     = A'*b;
y       = x0; 
u       = x;

obj         = @(x) .5*norm(A*x-b)^2 + lambda*(norm(x,1)-alpha*norm(x));
output.pm   = pm;

for it = 1:maxit                
    %update x
    x       = shrinkL12(y - u, lambda/delta, alpha);
    if it == 1
        x   = x0;
    end

    %update y
    yold    = y;
    rhs     = Atb  + delta * (x + u);
    y       = rhs/delta - (A'*(U\(L\(A*rhs))))/delta^2;
    
    %update u
    u       = u + x - y;
    
    % stop conditions & outputs
    relerr      = norm(yold - y)/max([norm(yold), norm(y), eps]);
    residual    = norm(A*x - b)/norm(b);
    
    output.relerr(it)   = relerr;
    output.obj(it)      = obj(x);
    output.time(it)     = toc(start_time);
    output.res(it)      = residual;
    output.err(it)      = norm(x - xg)/norm(xg);
    
    if relerr < reltol && it > 2
        break;
    end
end
end
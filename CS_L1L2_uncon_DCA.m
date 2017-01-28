function [x,output] = CS_L1L2_uncon_DCA(A,b,pm)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%         min_x .5||Ax-b||^2 + lambda(|x|_1 - alpha |x|_2)            %%%  
%%%                                                                     %%%
%%% Input: dictionary A, data b, parameters set pm                      %%%
%%%        pm.lambda: regularization paramter                           %%%
%%%        pm.maxoit: max outer DCA iterations                          %%%
%%%        pm.maxit: max inner ADMM iterations                          %%%
%%%        pm.delta: penalty parameter for ADMM                         %%%
%%%        pm.reltol: rel tolerance for ADMM: default value: 1e-6       %%%
%%% Output: computed coefficients x                                     %%%
%%%        output.relerr: relative error of yold and y                  %%%
%%%        output.obj: objective function of x_n:                       %%%
%%%        obj(x) = lambda(|x|_1 - |x|_2)+0.5|Ax-b|^2                   %%%
%%%        output.res: residual of x_n: norm(Ax-b)/norm(b)              %%%
%%%        output.err: error to the ground-truth: norm(x-xg)/norm(xg)   %%%
%%%        output.time: computational time                              %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



[M,N]       = size(A);  
start_time  = tic;

%% parameters
eps = 1e-16;

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
% maximum number of outer DCA iterations
if isfield(pm,'maxoit'); 
    maxoit = pm.maxoit; 
else 
    maxoit = 10; % default value
end
% maximum number of inner ADMM iterations
if isfield(pm,'maxit'); 
    maxit = pm.maxit; 
else 
    maxit = 2*N; % default value
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
AAt = A*A';
L = chol( speye(M) + 1/delta*AAt, 'lower' );
L = sparse(L);
U = sparse(L');

x = x0;
Atb = A'*b;
y = zeros(N,1); u = y;

obj = @(x) .5*norm(A*x-b)^2+ lambda*(norm(x,1)-alpha*norm(x));

kkk = 1;


for oit = 1:maxoit
    
    c = alpha*x/(norm(x,2)+eps); 
    xold = x;
    
    %ADMM method for solving the sub-problem
    for it = 1:maxit
            
            %update x
            xoldinner = x;
            x =shrink(y-u, lambda/delta);
            
            %update y
            yold = y;
            rhs = Atb + lambda* c + delta*(x+u);
            y = rhs/delta - (A'*(U\(L\(A*rhs))))/delta^2;

           
  
            %update u
            u = u + x-y;
    
            
            
            % stop conditions & outputs
            relerr = norm(x-y)/max([norm(x),norm(y),eps]);
            residual = norm(A*x-b)/norm(b);

    
               
            relerrinner = relerr;
            output.relerr(kkk) = relerrinner;
            output.obj(kkk) = obj(x);
            output.time(kkk) = toc(start_time);
            output.res(kkk) = residual;
            output.err(kkk) = norm(x-xg)/norm(xg);
            kkk = kkk+1;

            
        if relerr < reltol   
            break;
        end      
            
    end

    %Stopping condition for DCA
    relerr = sqrt(sum((x-xold).^2))/max(sqrt(sum(x.^2)),1);
    
    if relerr < reltol
        break;
    end
end


end




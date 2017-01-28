function [b,y,w,output] = construct_test4L12(A,x,lambda)


n = size(A,2);
s = length(find(x~=0));

maxIter = 10*n;
tolres = 1e-10;
verbose = false;

U = orth(A'); % Orthonormal basis for rg A'

w0 = zeros(n,1);
w0(x>0) = 1;
w0(x<0) = -1;
w = w0;

x2 = x/norm(x);

for iter=1:maxIter
    wold = w;
    v = U*(U'*(wold - x2)); % Projection onto rg A'
    error_rangeAt = norm(v-wold+x2);
    w = Ps(v,w0);
    %error_subgrad = norm(w - v);
    
    output.err(iter) = error_rangeAt;
    %output.err_sign(iter) = error_subgrad;
    if verbose
        fprintf('% 6d | pocs: distance to range: %4.2e\n',iter,error_rangeAt)
    end
    
    if error_rangeAt < tolres
       if verbose
           fprintf('pocs converged after %d iterations...\n',iter)
       end
       
       break;
    end
end



% Calculate b:
y = A'\(w-x2);
if verbose
    fprintf('A^T*y = w-x/norm(x) solved with residuum %e \n', norm(A'*y-w+x2))
end
b = lambda*y + A*x;
if verbose
    fprintf('b successfully constructed.\n\n')
end


function y = Ps(x,pattern)
% 
y = 0*x;
y(pattern > 0) = 1;
y(pattern < 0) = -1;
y(pattern== 0) = max(min(x(pattern==0),1),-1);
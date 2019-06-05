function x = shrinkL12(y,lambda,alpha)
% min_x .5||x-y||^2 + lambda(|x|_1- alpha |x|_2)

x = zeros(size(y));

if nargin<3
    alpha = 1;
end

if max(abs(y)) > 0 
    if max(abs(y)) > lambda;
        x   = max(abs(y) - lambda, 0).*sign(y);
        x   = x * (norm(x) + alpha * lambda)/norm(x);
    else
        if max(abs(y))>=(1-alpha)*lambda
            [~, i]  = max(abs(y));
            x(i(1)) = (abs(y(i(1))) + (alpha - 1) * lambda) * sign(y(i(1)));
        end
    end
end

return; 

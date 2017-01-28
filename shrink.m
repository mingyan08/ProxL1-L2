

function z = shrink(x, r)
    z = sign(x).*max(abs(x)-r,0);
end

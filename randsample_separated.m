function supp = randsample_separated(N,K,L)
% random sampling K integers from 1--N with spacing at least L 
supp = randsample(N-L*(K),K);
supp = sort(supp);
supp = supp + (0:K-1)'*L;
end
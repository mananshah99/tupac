function n=RieszNorm(Q)

n=0;
for iter=1:length(Q),
    tmp=abs(Q{iter}).^2;
    n=n+sum(tmp(:));
end;

n=sqrt(n);
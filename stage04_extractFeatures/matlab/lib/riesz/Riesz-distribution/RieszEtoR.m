function sub=RieszEtoR(orig,order)

% data conversion if needed
if ~iscell(orig),
    for iter=1:order+1,
        tmp{iter}=orig(:,:,iter);
    end;
    orig=tmp;
end;

if order==1,
    sub=orig;
    return;
end;

% load transformation
matfn=sprintf('RieszEtoR%d.dat',order);
mat=reshape(load(matfn),[order+1 order+1]);

for iter=1:order+1,
    sub{iter}=mat(iter,1)*orig{1};
    for iter2=2:order+1,
        sub{iter}=sub{iter}+mat(iter,iter2)*orig{iter2};
    end;
end

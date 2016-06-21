% (convention: positive angle = counterclockwise on Matlab image display)

function sub=RieszSteerOrder1(orig,th,order)

if ~iscell(orig),
%     if order==1,
%         tmp{1}=real(orig);
%         tmp{2}=imag(orig);
%     else
        for iter=1:order+1,
            tmp{iter}=orig(:,:,iter);
       end;
%     end;
    orig=tmp;
end;

% load steering matrix
matfn=sprintf('RieszSteer%d.dat',order);
mat=shiftdim(reshape(load(matfn),[order+1 order+1 order+1]),1);

% DEBUG: visualize steering matrix
if 0,
fprintf('\n');
for iter1=1:order+1, % row
    fprintf('[ '); 
    for iter2=1:order+1, % column
        for iter3=1:order+1, % term
            if mat(iter2,iter1,iter3),
                fprintf('%+3.1f cos(t)^%d sin(t)^%d ',mat(iter2,iter1,iter3),order+1-iter3,iter3-1);
            end;
        end;
        if iter2~=order+1,
            fprintf(', ');
        end;
    end;
    fprintf(']\n');
end;
end;

for iter=1:order+1,
    cs{iter}=cos(th).^(order-iter+1).*sin(th).^(iter-1);
end;
for iter1=1:order+1, % row
    sub{iter1}=zeros(size(orig{1},1),size(orig{1},2));
    for iter2=1:order+1, % column
        coef=zeros(size(sub{iter1}));
        for iter3=1:order+1, % term
            if mat(iter2,iter1,iter3),
                coef=coef+cs{iter3}.*mat(iter2,iter1,iter3);
            end;
        end;
        sub{iter1}=sub{iter1}+orig{iter2}.*coef;
    end;
end

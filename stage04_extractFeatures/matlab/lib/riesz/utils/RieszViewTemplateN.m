% Visualize template of Riesz derivative

function [I]=RieszViewTemplateN(orig,order,N,fig,titleTemplate)

if ~iscell(orig),
    if order==1,
        tmp{1}=real(orig);
        tmp{2}=imag(orig);
    else
        for iter=1:order+1,
            tmp{iter}=orig(iter);
       end;
    end;
    orig=tmp;
end;

% simulation grid (odd!)
% N=257;
% N=129;
%N=31;
% finite difference horizontal
Fx=zeros(N,N);
Fx(1,1)=1; Fx(1,end)=-1; 
Fx=Fx/sqrt(2);

% finite difference vertical
Fy=zeros(N,N);
Fy(1,1)=1; Fy(end,1)=-1;
Fy=Fy/sqrt(2);

% prepare filter
F=zeros(N,N);
for iter=0:order,
    F=F+orig{iter+1} ...
        * IterateDerivative(Fx,Fy,order-iter,iter) ...
        * sqrt(nchoosek(order,iter));
end;

I=fspecial('gaussian',[N N],N/15);
I=imfilter(I,fftshift(F),'circular');

if fig~=-1,
    if fig==1,
        figure;
    end;
    rng=max(abs(I(:)));
    imagesc(I,[-rng rng]);
    axis image;
    title(cast(titleTemplate, 'char'));
end;

function F=IterateDerivative(Fx,Fy,orderx,ordery)

if orderx>0,
    F=imfilter(Fx,fftshift(IterateDerivative(Fx,Fy,orderx-1,ordery)),'circular'); 
elseif ordery>0,
    F=imfilter(Fy,fftshift(IterateDerivative(Fx,Fy,orderx,ordery-1)),'circular');
else
    F=zeros(size(Fx));
    F(1,1)=1;
end;

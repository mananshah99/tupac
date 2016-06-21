function [th,mx]=RieszAngleTemplate(orig,template,order)

% data conversion if needed
if ~iscell(orig),
% disp('toCELL')
    if order==1,
        tmp{1}=real(orig);
        tmp{2}=imag(orig);
    else
        tmp=cell(1,order+1);
        for iter=1:order+1,
            tmp{iter}=orig(:,:,iter);
        end;
    end;
    orig=tmp;
end;

% load polynomial coefficients
matfn=sprintf('RieszAngle%d-Flex.dat',order); % contains formula for each
mat=reshape(load(matfn),[order+1 order+1 order+1]);
mat=shiftdim(mat,1);

% DEBUG: visualize polynomial
if 0,
fprintf('\n');
for iter1=1:order+1, % degree
    fprintf('tan(t)^%d ( ',order+1-iter1);
    for iter2=1:order+1, % channel
        fprintf('+ ch[%d] ( ',iter2);
        for iter3=1:order+1, % template
            if mat(iter3,iter2,iter1),
                fprintf('%+3.1f tm[%d] ',mat(iter3,iter2,iter1),iter3);
            end;
        end;
        fprintf(')\n    ');
    end;
    fprintf(')\n');
end;
end;

% load steering matrix
steermatfn=sprintf('RieszSteer%d.dat',order); 
steermat=shiftdim(reshape(load(steermatfn),[order+1 order+1 order+1]),1);

terms=cell(1,order+1);
for iter1=1:order+1, % degree
    terms{iter1}=zeros(size(orig{1},1),size(orig{1},2));
    for iter2=1:order+1, % channel
        for iter3=1:order+1, % template
            if mat(iter3,iter2,iter1),
                terms{iter1}=terms{iter1}+orig{iter2}.*template(iter3).*mat(iter3,iter2,iter1);
            end;
        end;
    end;
end

th=zeros(size(terms{1}));
mx=zeros(size(terms{1}));
tic;
for iterx1=1:size(terms{1},1),
    for iterx2=1:size(terms{1},2),
        C=zeros(1,order+1);
        for iter=1:order+1,
            C(iter)=terms{iter}(iterx1,iterx2);
        end;
        
        R=roots(C);        
        R=real(R(find(abs(imag(R))<1e-5)));
    
        if isempty(R),
            R=0;
        end;
       
        tha=atan(R); 
        
        
        costha=cos(tha); sintha=sin(tha);
        cossintha=cell(1,order+1);
        for iterterm=1:order+1,
            cossintha{iterterm}=costha.^(order-iterterm+1).*sintha.^(iterterm-1);          
        end;
        V = zeros(size(tha));
        for itertm=1:order+1,           % template (row)
            for iterch=1:order+1,       % channel (column)
                for iterterm=1:order+1, % term
                    V = V + ...
                        template(itertm)*steermat(itertm,iterch,iterterm) * ...
                        cossintha{iterterm} .* ...
                        orig{iterch}(iterx1,iterx2);
                end;
            end;
        end;         
        idx=find((V(:))==max((V(:)))); % before: abs!     
        th(iterx1,iterx2)=-tha(idx(1));         
        mx(iterx1,iterx2)=V(idx(1));
    end;
end;
toc;
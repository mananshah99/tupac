function [th,mx]=RieszAngleTemplateOptimized(orig,template,order)

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

for iter=1:order+1,
    termsVect(iter,:,:)=terms{iter}(:,:);
    origVect(iter,:,:)=orig{iter}(:,:);
end
th=zeros(size(terms{1}));
mx=zeros(size(terms{1}));

% tic;
idx1=1:size(terms{1},1);
idx2=1:size(terms{1},2);

C=zeros(order+1,size(idx1,2),size(idx2,2));
for iter=1:order+1,
    C(iter,idx1,idx2)=termsVect(iter,idx1,idx2);
end;
Rmat=nan(order,size(terms{1},1),size(terms{1},2));
% tic;
for iterx1=idx1,
    for iterx2=idx2,
        R=roots(C(:,iterx1,iterx2));
         
        R=real(R(abs(imag(R))<1e-5));
        if isempty(R),
            R=0;
        end;
        Rmat(1:size(R,1),iterx1,iterx2)=R; 
    end;
end;
% toc;
tha=atan(Rmat); 
costha=cos(tha); sintha=sin(tha);
cossintha=zeros(order+1,size(costha,1),size(idx1,2),size(idx2,2));
for iterterm=1:order+1,
    cossintha(iterterm,:,idx1,idx2)=costha(:,idx1,idx2).^(order-iterterm+1).*sintha(:,idx1,idx2).^(iterterm-1);
end;
V = zeros(size(tha,1),size(idx1,2),size(idx2,2));
for itertm=1:order+1,           % template (row)
    for iterch=1:order+1,       % channel (column)
        for iterterm=1:order+1, % term
            coeff=template(itertm)*steermat(itertm,iterch,iterterm)*origVect(iterch,idx1,idx2);
            sumterm=repmat(coeff,[size(squeeze(cossintha(iterterm,:,idx1,idx2)),1) 1 1]).*squeeze(cossintha(iterterm,:,idx1,idx2));
            V(:,idx1,idx2)=V(:,idx1,idx2)+sumterm;
        end;
    end;
end;
maxV(idx1,idx2)=max(V(:,idx1,idx2));
maxV=shiftdim(repmat(maxV,[1 1 size(V,1)]),2);
idxV=V-maxV+1;%idee: soutraire V à maxV: les zeros donneront l'index
idxV(idxV<1)=0;
idxV(isnan(idxV))=0;tha(isnan(tha))=0;V(isnan(V))=0;

th=squeeze(sum(-tha.*idxV,1));
mx=squeeze(sum(V.*idxV,1));

% toc;
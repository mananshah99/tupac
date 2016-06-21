function [th,mx]=RieszAngle(orig,order,varargin)

% data conversion if needed
if ~iscell(orig),
    if order==1,
        tmp{1}=real(orig);
        tmp{2}=imag(orig);
    else
        for iter=1:order+1,
            tmp{iter}=orig(:,:,iter);
        end;
    end;
    orig=tmp;
end;

if ~isempty(varargin), % channel available?
    channel=varargin{1};
    if channel>floor(order/2+1),
        error('[RieszAngle] Channel argument exceeds maximum');
    end;
else
    channel=1;
end;

template=zeros(1,order+1);
template(channel)=1;

[th,mx]=RieszAngleTemplate(orig,template,order);

% 
% % load polynomial coefficients
% matfn=sprintf('RieszAngle%d-Ch%d.dat',order,channel);
% mat=reshape(load(matfn),[order+1 order+1]);
% %mat=shiftdim(mat,1);
% 
% % DEBUG: visualize polynomial
% if 1,
% fprintf('\n');
% for iter1=1:order+1, % degree
%     fprintf('tan(t)^%d ( ',order+1-iter1);
%     for iter2=1:order+1, % channel
%         fprintf('%+3.1f ch[%d] ',mat(iter2,iter1),iter2);
%     end;
%     fprintf(')\n');
% end;
% end;
% 
% 
% % load steering matrix
% steermatfn=sprintf('RieszSteer%d.dat',order);
% steermat=shiftdim(reshape(load(steermatfn),[order+1 order+1 order+1]),1);
% 
% for iter1=1:order+1,
%     terms{iter1}=zeros(size(orig{1},1),size(orig{1},2));
%     for iter2=1:order+1,
%         if mat(iter2,iter1),
%             terms{iter1}=terms{iter1}+orig{iter2}.*mat(iter2,iter1);
%         end;
%     end;
% end
% 
% th=zeros(size(terms{1}));
% mx=zeros(size(terms{1}));
% for iterx1=1:size(terms{1},1),
%     for iterx2=1:size(terms{1},2),
%         for iter=1:order+1,
%             C(iter)=terms{iter}(iterx1,iterx2);
%         end;
%         R=real(roots(C(find(abs(imag(C))<1e-5))));
%         if isempty(R),
%             R=[0];
%         end;
%         tha=atan(R);
%         V = zeros(size(tha));
%         for iterch=1:order+1,       % channel
%             for iterterm=1:order+1, % term
%                 V = V + ...
%                     steermat(channel,iterch,iterterm) * ...
%                     cos(tha).^(order-iterch+1).*sin(tha).^(iterch-1) .* ...
%                     orig{iterch}(iterx1,iterx2);
%             end;
%         end;
%         idx=find(abs(V(:))==max(abs(V(:))));
%         th(iterx1,iterx2)=-tha(idx(1));
%         mx(iterx1,iterx2)=max(abs(V(:)));
%     end;
% end;

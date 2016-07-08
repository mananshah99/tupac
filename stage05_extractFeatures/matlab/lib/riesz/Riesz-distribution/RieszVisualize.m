function I=RieszVisualize(Q,config,J)

tmpN=config.N;

for iter=1:config.riesz.channels,
    Q0=Q;
    for iter2=1:J+1,
        if tmpN==1, % & 0 % complex!
            if iter==1,
                Q0{iter2}=real(Q0{iter2});
                fprintf('complex');
            else
                Q0{iter2}=imag(Q0{iter2});
            end;
        else
            Q0{iter2}=Q0{iter2}(:,:,iter);
        end;
    end;
    config.N=0;
    I{iter}=visualizePyramid(Q0,config,J);
    subplot(1,config.riesz.channels,iter);
    imshow(I{iter},[]);
    set(gcf,'Color',[1 1 1]);
end;

function I=visualizePyramid(Q,config,J);

DLM=5;

if config.N>0,
    CMPLX=1;
else
    CMPLX=0;
end;

l1=1;
l2=1;
r1=1;
r2=size(Q{1},2)+DLM;

for iter=1:J+1,
    s1=size(Q{iter},1)-1;
    s2=size(Q{iter},2)-1;
    if iter<J+1,
        s2n=s2-(size(Q{iter+1},2)-1);
    else
        s2n=(s2+1)/2;
    end;
    %Q{iter}(find(Q{iter}>1e4))=1e4;
    Q{iter}=Q{iter}-min(abs(Q{iter}(:)));
    Q{iter}=Q{iter}/(10*eps+max(abs(Q{iter}(:))));
    if CMPLX==0,
        I(l1:l1+s1,l2:l2+s2)=real(Q{iter})+1e-10;
    else
    if iter<J+1,
        tmp2=-real(Q{iter});
        tmp1=imag(Q{iter});
    else
        tmp1=real(Q{iter});
        tmp2=imag(Q{iter});
    end;
    I(l1:l1+s1,l2:l2+s2)=tmp1;
    I(r1:r1+s1,r2:r2+s2)=tmp2;
    end;
    l1=l1+s1+DLM;
    l2=l2+s2n;
    r1=r1+s1+DLM;
end;

%I(find(abs(I)<10*eps))=Inf;
I(find(abs(I)==0))=Inf;


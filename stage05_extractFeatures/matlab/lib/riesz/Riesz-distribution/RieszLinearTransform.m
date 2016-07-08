function Q=RieszLinearTransform(Q,A,order)

% rearrange
for iter=1:order+1,
    tmp=Q{iter};
    QR(iter,:)=tmp(:);
end;

QR=A.'*QR;

% back arrange
for iter=1:order+1,
    Q{iter}=reshape(QR(iter,:),size(Q{iter}));
end;



function Q=RieszExpand(F,Qex)

Q=Qex;

for iter=1:length(Q),
    tmp=F(1:length(Q{iter}(:)));
    F=F(length(Q{iter}(:))+1:end);
    Q{iter}(:)=tmp(:);
end;

    
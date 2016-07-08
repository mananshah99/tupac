function F=RieszFlatten(Q);

F=[];

for iter=1:length(Q),
    tmp=Q{iter};
    F=[F; tmp(:)];
end;

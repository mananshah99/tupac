function LP=simoncelli_synthesis(LP,HP,config,step)

if size(LP,3)>1,
    for iter=1:size(LP,3),
        tmp(:,:,iter)=simoncelli_synthesis(LP(:,:,iter),HP(:,:,iter),config,step);
    end;
    LP=tmp;
else
    LP0=zeros(2*size(LP));
    LP0(1:2:end,1:2:end)=LP;
    FLP=fft2(LP0);
    FLP=FLP.*config.filter.lowpass(1:step:end,1:step:end);
    LP=ifft2(FLP);
    FHP=fft2(HP);
    FHP=FHP.*config.filter.highpass(1:step:end,1:step:end);
    LP=LP+ifft2(FHP); 

    if strcmpi(config.datatype,'real'),
        LP=real(LP);
    end;
end;

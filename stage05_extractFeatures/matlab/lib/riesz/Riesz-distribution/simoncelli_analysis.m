function [LP,HP]=simoncelli_analysis(I,config,step)

if size(I,3)>1,
    for iter=1:size(I,3),
        [LP(:,:,iter),HP(:,:,iter)]=simoncelli_analysis(I(:,:,iter),config,step);
    end;
else

    FI=fft2(I);
    FLP=FI.*config.filter.lowpass(1:step:end,1:step:end);
    LP=ifft2(FLP);
    LP=LP(1:2:end,1:2:end);
    HP=FI.*config.filter.highpass(1:step:end,1:step:end);
    
    HP=ifft2(HP);
    
    if strcmpi(config.datatype,'real'),
        LP=real(LP);
        HP=real(HP);
    end;
end;

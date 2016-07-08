function [LP,HP]=simoncelli_analysis_initial(I,config)

if size(I,3)>1,
    for iter=1:size(I,3),
        [LP(:,:,iter),HP(:,:,iter)]=simoncelli_analysis_initial(I(:,:,iter),config);
    end;
else
    FI=fft2(I);
    LP=FI.*config.filter.initiallowpass;
    LP=ifft2(LP);
    HP=FI.*config.filter.initialhighpass;
    HP=ifft2(HP);
    
    if strcmpi(config.datatype,'real'),
        LP=real(LP);
        HP=real(HP);
    end;
end;

function I=simoncelli_synthesis_initial(LP,HP,config)

if size(LP,3)>1,
    for iter=1:size(LP,3),
        I(:,:,iter)=simoncelli_synthesis_initial(LP(:,:,iter),HP(:,:,iter),config);
    end;
else
    FLP=fft2(LP);
    FHP=fft2(HP);
    FLP=FLP.*config.filter.initiallowpass;
    I=ifft2(FLP);
    FHP=FHP.*config.filter.initialhighpass;
    HP=ifft2(FHP);
    I=I+HP;
    
    if strcmpi(config.datatype,'real'),
        I=real(I);
    end;
end;

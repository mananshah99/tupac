function LP=simoncelli_synthesis_redundant(LP,HP,config,step,iterScale)

    if size(LP,3)>1,
        for iter=1:size(LP,3),
            tmp(:,:,iter)=simoncelli_synthesis_redundant(LP(:,:,iter),HP(:,:,iter),config,step,iterScale);
        end;
        LP=tmp;
    else

    
    LP0=LP;
    FLP=fft2(LP0);
    
    sizeInit=size(config.filter.lowpass,1);
    sizeDown=size(config.filter.lowpass(1:step:end,1:step:end),1);
        
    if iterScale~=1
        energyFactor=1/2;
    else
        energyFactor=1;
    end;
        
    lowpass=config.filter.lowpass(1:step:end,1:step:end).*energyFactor;
    lowpass=[lowpass(1:floor(end/2),:);zeros(sizeInit-sizeDown,sizeDown);lowpass(floor(end/2)+1:end,:)];
    lowpass=[lowpass(:,1:floor(end/2)) zeros(sizeInit,sizeInit-sizeDown) lowpass(:,floor(end/2)+1:end)];
    
    FLP=FLP.*lowpass;
    LP=ifft2(FLP);
    FHP=fft2(HP);
    
    sizeInit=size(config.filter.highpass,1);
    sizeDown=size(config.filter.highpass(1:step:end,1:step:end),1);
    
    highpass=config.filter.highpass(1:step:end,1:step:end).*energyFactor;
    highpass=[highpass(1:floor(end/2),:);ones(sizeInit-sizeDown,sizeDown);highpass(floor(end/2)+1:end,:)];
    highpass=[highpass(:,1:floor(end/2)) ones(sizeInit,sizeInit-sizeDown) highpass(:,floor(end/2)+1:end)];
    FHP=FHP.*highpass;
    LP=LP+ifft2(FHP); 

    if strcmpi(config.datatype,'real'),
        LP=real(LP);
    end;
end;

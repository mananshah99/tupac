function [LP,HP]=simoncelli_analysis_redundant(I,config,step,iterScale)

    if size(I,3)>1,
        for iter=1:size(I,3),
            [LP(:,:,iter),HP(:,:,iter)]=simoncelli_analysis_redundant(I(:,:,iter),config,step,iterScale);
        end;
    else

    FI=fft2(I);
    
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
    FLP=FI.*lowpass;
    LP=ifft2(FLP);
    
    highpass=config.filter.highpass(1:step:end,1:step:end).*energyFactor;
    maxValHighpass=max(max(highpass));
    highpass=[highpass(1:floor(end/2),:);ones(sizeInit-sizeDown,sizeDown)*maxValHighpass;highpass(floor(end/2)+1:end,:)];
    highpass=[highpass(:,1:floor(end/2)) ones(sizeInit,sizeInit-sizeDown)*maxValHighpass highpass(:,floor(end/2)+1:end)];
    HP=FI.*highpass;
    HP=ifft2(HP);
    
    if strcmpi(config.datatype,'real'),
        LP=real(LP);
        HP=real(HP);
    end;
end;

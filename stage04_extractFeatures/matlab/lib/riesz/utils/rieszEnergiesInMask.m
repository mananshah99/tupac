function energies=rieszEnergiesInMask(riesz,mask,pyramid)

mask=double(mask);
energies=[];
energyFactor=1;
for iterScale=1:size(riesz,2)-2,
    idxNonZeros=find(mask~=0);
    parfor iterRiesz=1:size(riesz{1},3),
        tmp=riesz{iterScale};
        subband=tmp(1:end,1:end,iterRiesz);
        k=size(idxNonZeros,1);
        c=subband(idxNonZeros);
        energy=sqrt(sum(c.^2)/k)*energyFactor;
        energies=[energies energy];
    end;
    if pyramid,
        mask=mask(1:2:end,1:2:end);
    else
        energyFactor=energyFactor*2;
    end;
end;
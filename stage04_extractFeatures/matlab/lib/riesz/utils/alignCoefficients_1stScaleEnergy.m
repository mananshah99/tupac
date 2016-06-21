function learnedFeatures = alignCoefficients_1stScaleEnergy(rieszCoeffs,w,N,J)

learnedFeatures=[];

% align Riesz components based on dominant orientation of the signature w (1st scale)
if N==1,
    [theta,~]=RieszAngleTemplateOrder1(rieszCoeffs{1},w(1:N+1),N);
else
    [theta,~]=RieszAngleTemplateOptimized(rieszCoeffs{1},w(1:N+1),N);
end;

% align coefficients
tmpScales=cell(J,1);
for countScale=1:J,
    tmpScales{countScale}=RieszSteerOrder1(rieszCoeffs{countScale},theta,N);
    theta=theta(1:2:end,1:2:end);
end;

for countScale=1:J,
    for iterRiesz=1:N+1,
        tmp=tmpScales{countScale};
        subband=tmp{1,iterRiesz}(:,:); 
        [m,n]=size(subband);
        k=m*n;
        c=subband(:);
        l=sqrt(sum(c.^2)/k);
        learnedFeatures=[learnedFeatures l];
    end;
end;
end

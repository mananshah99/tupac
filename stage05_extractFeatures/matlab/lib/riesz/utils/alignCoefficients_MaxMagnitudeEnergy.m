function learnedFeaturesMaxMagnitude = alignCoefficients_MaxMagnitudeEnergy(rieszCoeffs,w,N,J)

learnedFeaturesMaxMagnitude=[];

countScale=0;
for iterScale=1:N+1:J*(N+1),

    wS=w(iterScale:iterScale+N);
    countScale=countScale+1;

    % align Riesz components based on dominant orientation of the signature w
    if N==1,
        [~,magnitude]=RieszAngleTemplateOrder1(rieszCoeffs{countScale},wS,N);
    else
        [~,magnitude]=RieszAngleTemplateOptimized(rieszCoeffs{countScale},wS,N);
    end;

    [m,n]=size(magnitude);
    k=m*n;
    c=magnitude(:);
    l=sqrt(sum(c.^2)/k);
    learnedFeaturesMaxMagnitude=[learnedFeaturesMaxMagnitude l]; 
end;   
end


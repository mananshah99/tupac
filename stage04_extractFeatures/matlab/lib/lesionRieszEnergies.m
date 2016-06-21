function res=lesionRieszEnergies(image,mask,N,J)
   
    % parameters
    pyramid=false;
    align=false;

    rieszCoeffs = RieszTextureAnalysis(image,N,J,align,pyramid);
    res = rieszEnergiesInMask(rieszCoeffs,mask,pyramid);
       
end
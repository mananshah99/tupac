% Stanford, 10/10/2013
% -------------------------------------------------------------------
% demoRieszTextureAnalysis2D.m
% 
% Demo script for Riesz texture analysis
% -------------------------------------------------------------------
%
% Adrien Depeursinge, adrien.depeursinge@hevs.ch
% Stanford University, CA, USA
%
% -------------------------------------------------------------------
%
% REFERENCES:
% [1] A. Depeursinge, A. Foncubierta-Rodriguez, D. Van De Ville, H. Müller, 
%     "Rotation–covariant texture learning using steerable Riesz wavelets",
%     in: IEEE Transactions on Image Processing,(submitted)
%
% [2] A. Depeursinge, A. Foncubierta-Rodriguez, D. Van De Ville, H. Müller,
%     "Lung texture classification using locally-oriented Riesz components",
%     in: Medical Image Computing and Computer-Assisted Intervention –
%     MICCAI 2011, Toronto, Canada, pages 231-238, Springer Berlin / Heidelberg, 2011 
%
% [3] M. Unser, D. Van De Ville, N. Chenouard
%     "Steerable Pyramids and Tight Wavelet Frames in L2(Rd)",
%     in: IEEE Transactions on Image Processing, 20:10(2705-2721), 2011
%
% -------------------------------------------------------------------

clc;close all;clear all;

% Create toy image
imageSize=128;
[X,Y] = meshgrid(1:imageSize,1:imageSize);
I1=sin(X/(imageSize/96))+sin(Y/(imageSize/96))+rand(imageSize);
imshow(I1,[]);

% create toy mask
mask(imageSize/4:3*imageSize/4,imageSize/4:3*imageSize/4)=1;

rieszOrder=2;     % Riesz order
numberOfScales=floor(log2(imageSize))-1; % number of decomposition levels
pyramid=true; % pyramid wavelet decomposition VS undecimated
align=false;  % maximize the response of R^(0,N) (see [1])

% compute Riesz coefficients
rieszCoeffs=RieszTextureAnalysis(I1,rieszOrder,numberOfScales,align,pyramid);
% viewRieszCoefficient(rieszCoeffs);
energies=rieszEnergiesInMask(rieszCoeffs,mask,pyramid)

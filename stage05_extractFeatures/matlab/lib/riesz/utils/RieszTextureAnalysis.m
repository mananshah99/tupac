% Stanford, 10/10/2013
%-------------------------------------------------------------------
%
% Adrien Depeursinge, adrien.depeursinge@hevs.ch
% Stanford University, CA, USA
%
% -------------------------------------------------------------------
%
% REFERENCES:
% [1] A. Depeursinge, A. Foncubierta-Rodriguez, D. Van De Ville, H. M?ller, 
%     "Rotation?covariant texture learning using steerable Riesz wavelets",
%     in: IEEE Transactions on Image Processing,(submitted)
%
% [2] A. Depeursinge, A. Foncubierta-Rodriguez, D. Van De Ville, H. M?ller,
%     "Lung texture classification using locally-oriented Riesz components",
%     in: Medical Image Computing and Computer-Assisted Intervention ?
%     MICCAI 2011, Toronto, Canada, pages 231-238, Springer Berlin / Heidelberg, 2011 
%
% [3] M. Unser, D. Van De Ville, N. Chenouard
%     "Steerable Pyramids and Tight Wavelet Frames in L2(Rd)",
%     in: IEEE Transactions on Image Processing, 20:10(2705-2721), 2011
%
% -------------------------------------------------------------------

function rieszCoeffs=RieszTextureAnalysis(I, N, J, align, pyramid)

% Configure Riesz transform              
clear config;
config.size=size(I);                 % size of data
config.datatype='real';              % 'real', 'complex'
config.type='Simoncelli';            % 'bspline', 'dual', 'ortho', 'Simoncelli' (for B-spline: at synthesis side)
config.Simoncelli.initial=false;     % perform initial high-pass filter or not
config.N=N;                          % order of the Riesz transform
config.alpha=1;                      % degree of B-spline pyramid (integer)
config.prefilter.type='bandlimited'; % 'bandlimited', 'none'
config.downsampling=pyramid;         % 0:redundant transform

 % Precompute filters
config=RieszPrepareTransform(config);
if ~config.ready,
    error('Riesz-Pyramid decomposition not ready!');
end;
                    
% Prefilter (projection into initial approximation space V0)
projected = RieszPrefilter(I,config);
% perform Riesz transform
rieszCoeffs = RieszAnalysis(projected,config,J);

if align,
    
    % if N=1, complexify:
    if config.N==1,
        for iterScale=1:J+1,
            rieszCoeffs{iterScale}=(rieszCoeffs{iterScale}(:,:,1))+j*(rieszCoeffs{iterScale}(:,:,2));
        end;
    end;

    % Align Riesz Components based on dominant orientation of h1 (Eq. 4 of [2])
    template=[1 zeros(1,N)];
    if config.N==1,
        [theta,~]=RieszAngleTemplate(rieszCoeffs{1},template,N);
    else
        [theta,~]=RieszAngleTemplateOptimized(rieszCoeffs{1},template,N);
    end;

    % align coefficients
    tmpScales=cell(J,1);
    for countScale=1:J,
        tmpScales{countScale}=RieszSteer(rieszCoeffs{countScale},theta,N);
        tmp=tmpScales{countScale};
        for iterRiesz=1:N+1,
            rieszCoeffs{countScale}(1:end,1:end,iterRiesz)=tmp{1,iterRiesz}(:,:);
        end;
        if pyramid==1,
            theta=theta(1:2:end,1:2:end);
        end;
    end;
end;


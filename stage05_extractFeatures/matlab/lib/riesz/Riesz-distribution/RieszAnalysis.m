function Q=RieszAnalysis(A,config,J)

% Riesz transform
if config.N>0,
    FA=fft2(A);
    for iter=1:config.riesz.channels,
      A(:,:,iter)=ifft2(FA.*config.riesz.filter(:,:,iter));
    end;
end;

if strcmpi(config.type,'simoncelli'), % Simoncelli pyramid
    Q=cell(1,J+2);
    if config.Simoncelli.initial,
        [C0,Q{J+2}]=simoncelli_analysis_initial(A,config);
    else
        C0=A;
    end;
    step=1;
    

    for iter=1:J,

        if config.downsampling==1,
            [C1,C2]=simoncelli_analysis(C0,config,step);
            Q{iter}=C2;
            C0=C1;
        else
            [C1,C2]=simoncelli_analysis_redundant(C0,config,step,iter);
            Q{iter}=C2;
            C0=C1;
        end;
        
        step=step*2;
    end;
    if config.downsampling==1,
        Q{J+1}=C0;
    else
        Q{J+1}=C0./2;
    end;
    
else % Spline pyramid
    step=1;
    C0=A;
    for iter=1:J,
        C1=ModuleREDUCE(C0,config.filter.analysis(1:step:end,1:step:end));
        C2=ModuleEXPAND(C1,config.filter.synthesis(1:step:end,1:step:end));
        
        % store detail coefficients
        Q{iter}=C0-C2;
        
        % prepare for next iteration
        C0=C1;
        step=step*2;
    end;    
    Q{J+1}=C0;
end;

% if real:
if strcmp(lower(config.datatype),'real'),
    for iter=1:J+1,
        Q{iter}=real(Q{iter});
    end;
end;

% if N=1, complexify:
% if config.N==1,
%     for iter=1:J+1,
%         Q{iter}=(Q{iter}(:,:,1))+j*(Q{iter}(:,:,2));
%     end;
% end;

% figure;
% subplot(2,2,2);
% imshow(A(:,:,1)); title('d^2/dx^2');
% subplot(2,2,3);
% imshow(A(:,:,2)); title('d^2/dxdy');
% subplot(2,2,4);
% imshow(A(:,:,3)); title('d^2/dy^2');

function C1=RieszSynthesis(Q,config,J);

% if N=1, complexify:
if config.N==1,
    for iter=1:J+1,
        tmp=Q{iter};
        Q{iter}(:,:,1)=real(tmp);
        Q{iter}(:,:,2)=imag(tmp);
    end;
end;

if strcmpi(config.type,'simoncelli'), % Simoncelli pyramid
    step=2^(J-1);
    
    if config.downsampling==1,
        C1=Q{J+1};
    else
        C1=Q{J+1}./2;
    end;

    for iter=J:-1:1,
        
        C2=Q{iter};
        
        if config.downsampling==1,
            C1=simoncelli_synthesis(C1,C2,config,step);
        else
            C1=simoncelli_synthesis_redundant(C1,C2,config,step,iter);
        end;
        step=step/2;
    end;
    if config.Simoncelli.initial
        C1=simoncelli_synthesis_initial(C1,Q{J+2},config);
    end;
else
    C1=Q{J+1};
    step=2^(J-1);
    for iter=J:-1:1,
        C0 = ModuleEXPAND( ...
            C1 - ModuleREDUCE( ...
            Q{iter}, ...
            config.filter.analysis(1:step:end,1:step:end)), ...
            config.filter.synthesis(1:step:end,1:step:end)) ...
            +Q{iter};
        
        % prepare for next iteration
        C1=C0;
        step=step/2;
    end;
end;

% Riesz transform
if config.N>0,
    FA=fft2(C1);
    C1=zeros(config.size);
    for iter=1:config.riesz.channels,
      C1=C1+ifft2(FA(:,:,iter).*conj(config.riesz.filter(:,:,iter)));
    end;
end;


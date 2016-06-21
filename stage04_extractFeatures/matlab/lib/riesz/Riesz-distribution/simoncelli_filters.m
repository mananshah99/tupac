function config = simoncelli_filters(config)

twidth=1;

dims = config.size;
ctr = ceil((dims+0.5)/2);

[xramp,yramp] = meshgrid( ([1:dims(2)]-ctr(2))./(dims(2)/2), ...
    ([1:dims(1)]-ctr(1))./(dims(1)/2) );
angle = atan2(yramp,xramp);
log_rad = sqrt(xramp.^2 + yramp.^2);
log_rad(ctr(1),ctr(2)) =  log_rad(ctr(1),ctr(2)-1);
log_rad  = log2(log_rad);

%% Radial transition function (a raised cosine in log-frequency):
[Xrcos,Yrcos] = simoncelli_rcosFn(twidth,(-twidth/2),[0 1]);
Yrcos = sqrt(Yrcos);

YIrcos = sqrt(1.0 - Yrcos.^2);
lo0mask = simoncelli_pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
config.filter.initiallowpass = ifftshift(lo0mask);

hi0mask = simoncelli_pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
config.filter.initialhighpass = ifftshift(hi0mask);

%% Subband filter (high-pass)
Xrcos = Xrcos - log2(2);  % shift origin of lut by 1 octave.
himask = simoncelli_pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
config.filter.highpass = ifftshift(himask);

%% Subband filter (low-pass)
%dims = size(lodft);
ctr = ceil((dims+0.5)/2);
lodims = ceil((dims-0.5)/2);
loctr = ceil((lodims+0.5)/2);
lostart = ctr-loctr+1;
loend = lostart+lodims-1;

%log_rad = log_rad(lostart(1):loend(1),lostart(2):loend(2));
YIrcos = abs(sqrt(1.0 - Yrcos.^2));
lomask = simoncelli_pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
config.filter.lowpass = ifftshift(lomask)*2;

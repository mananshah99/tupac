% -------------------------------------------------------------------
% RieszPrepareTransform.m
% 
% Prepare filters for Riesz pyramid
% -------------------------------------------------------------------
%
% Dimitri Van De Ville
% Ecole Polytechnique F?d?rale de Lausanne
%
% dimitri.vandeville@epfl.ch
%
% -------------------------------------------------------------------

function config=RieszPrepareTransform(config);

config.ready=0;

% Check validity of Riesz pyramid
if config.alpha<0 | config.alpha~=round(config.alpha),
    disp('Riesz-Pyramid requires integer-degree B-splines');
    return;
end;

% Prepare filters
[config.grid.xo,config.grid.yo]=ndgrid(2*pi*([1:config.size(1)]-1)/config.size(1),2*pi*([1:config.size(2)]-1)/config.size(2)); 
[config.grid.x2,config.grid.y2]=ndgrid(2*pi*([1:config.size(1)*2]-1)/(config.size(1)*2),2*pi*([1:config.size(2)*2]-1)/(config.size(2)*2)); 

config.grid.xc=config.grid.xo-pi;
config.grid.yc=config.grid.yo-pi;

tau=0;
if config.alpha==0,
    tau=1/2; % causal
end;

if strcmpi(config.type,'simoncelli'),
    config=simoncelli_filters(config);
    config.prefilter.filter=ones(config.size);
else    
    [FA,FS,A]=make2Dfilters(config.size(1),config.size(2),config.alpha,tau,config.type);
    
    config.filter.analysis=conj(FA);
    config.filter.synthesis=FS;
    config.filter.autocorr=A;
    
    config=make2Dprefilter(config,tau);
end;

if config.N>0, % enable Riesz transform 
    tmp=sqrt(config.grid.xc.^2+config.grid.yc.^2);
    config.riesz.channels=nchoosek(2+config.N-1,config.N);
    base1=-1i*config.grid.yc./tmp; base1=ifftshift(base1); base1(1,1)=1; 
    base2=-1i*config.grid.xc./tmp; base2=ifftshift(base2); base2(1,1)=1;
    base2(end/2+1,:)=imag(base2(end/2+1,:));
    base1(:,end/2+1)=imag(base1(:,end/2+1));
    
    for iter=1:config.riesz.channels,
        config.riesz.filter(:,:,iter) = sqrt(multinomial(config.N,config.riesz.channels-iter,iter-1)) * ...
            base1.^(config.riesz.channels-iter) .* ...
            base2.^(iter-1);
    end;
    
    config.riesz.filter(1,1,:)=config.riesz.filter(1,1,:)/sqrt(2^config.N);
else
    config.riesz.channels=1;
end;

config.ready=1;

%======= AUXILIARY FUNCTIONS =========

function config=make2Dprefilter(config,tau)

switch lower(config.prefilter.type),
    case 'none',
        config.prefilter.filter=ones(config.size);
    case 'bandlimited',
        config.prefilter.filter=bspline_sinc(config.grid.xc).*bspline_sinc(config.grid.yc);
        if config.alpha==0,
            config.prefilter.filter=config.prefilter.filter.*exp(-j*config.grid.xc*tau).*exp(-j*config.grid.yc*tau);
            config.prefilter.filter(1,:)=config.prefilter.filter(1,:).*exp(j*config.grid.xc(1,:)*tau).*exp(j*config.grid.yc(1,:)*tau);
            config.prefilter.filter(:,1)=config.prefilter.filter(:,1).*exp(j*config.grid.xc(:,1)*tau).*exp(j*config.grid.yc(:,1)*tau);
        else
            config.prefilter.filter=config.prefilter.filter.^(config.alpha+1);
        end;
        config.prefilter.filter=ifftshift(config.prefilter.filter);
        switch lower(config.type(1)),
            case 'o',
                config.prefilter.filter=config.prefilter.filter./sqrt(config.filter.autocorr);
            case 'b',
                config.prefilter.filter=config.prefilter.filter./config.filter.autocorr;
            case 'd', % OK
        end;
end;


function y=bspline_sinc(x)
%SINC: Sin(x/2)/(x/2) function.
i=find(x==0);                                                              
x(i)= 1;                           
y = sin(x/2)./(x/2);                                                     
y(i) = 1;   


function F=make_real(F0)
F=fft2(real(ifft2(F0)));
for iter=1:5,
    F(1:end/2,1:end/2)=F0(1:end/2,1:end/2);
    F(1:end/2,end/2+2:end)=F0(1:end/2,end/2+2:end);
    F(end/2+2:end,1:end/2)=F0(end/2+2:end,1:end/2);
    F(end/2+2:end,end/2+2:end)=F0(end/2+2:end,end/2+2:end);
    F(end/2+1,:)=abs(F0(end/2+1,:)).*sign(real(F(end/2+1,:)));
    F(:,end/2+1)=abs(F0(:,end/2+1)).*sign(real(F(:,end/2+1)));
    F=fft2(real(ifft2(F)));
end;

function [FA,FS,A]=make2Dfilters(M1,M2,alpha,tau,type);

[FA1,FS1,A1]=FFTfractsplinefilters(M1,alpha,tau,type);
[FA2,FS2,A2]=FFTfractsplinefilters(M2,alpha,tau,type);

FA(:,:)=FA1(1,:).'*FA2(1,:);
FS(:,:)=FS1(1,:).'*FS2(1,:);
A(:,:)=A1(1,:).'*A2(1,:);

function [FFTanalysisfilters,FFTsynthesisfilters,A]=FFTfractsplinefilters(M,alpha,tau,type)
%	Usage: [FFTanalysisfilters,FFTsynthesisfilters,A]=FFTfractsplinefilters(M,alpha,tau,type) 
%
% 	Provides the frequency response of the lowpass filters that generate the 
%   orthonormal or semi-orthonormal (B-spline or dual) fractional splines of 
%   degree alpha, shift tau and of given type.
%
% 	Author: Thierry Blu, October 1999, Revised January 2001
% 	Biomedical Imaging Group, EPFL, Lausanne, Switzerland.
%
%   Adapted for pyramid-style decomposition
%   Dimitri Van De Ville, September 2008
%
%		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%		%%%%%%%%%%%% INPUT %%%%%%%%%%%%	
%		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  M 	: size of the input signal = length of FFTfilters = 2^N
%  alpha 	: degree of the fractional splines, must be >-0.5 
%  tau	: shift or asymmetry of the fractional splines, we suggest to restrict this value to
%		the interval [-1/2,+1/2] because tau+1 leads to the same wavelet space as tau.
%		Particular cases
%		 	tau=0 <=> symmetric splines; 
%			tau=?1/2 <=> max. dissymetric splines
%   			tau=(alpha+1)/2 <=> causal splines)
%  type	: type of the B-splines
%		= 'ortho' (orthonormal, default)
% 		= 'bspline' (B-spline) 
% 		= 'dual' (dual). The last option is the flipped version of the B-spline one.
%
%		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
%		%%%%%%%%%%%% OUTPUT %%%%%%%%%%%	
%		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
%
% 	FFTanalysisfilters	= [lowpassfilter]	: FFT filter arrays
% 	FFTsynthesisfilters	= [lowpassfilter] 	: FFT filter arrays
%   A (autocorrelation)                     : FFT filter arrays
%   
% 
%		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
%		%%%%%%%%%%%% REFERENCES %%%%%%%
%		%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%	
%
% 	References:
% 	[1] M. Unser and T. Blu, "Fractional splines and wavelets," 
% 	SIAM Review, Vol. 42, No. 1, pp. 43--67, January 2000.
% 	[2] M. Unser and T. Blu, "Construction of fractional spline wavelet bases," 
% 	Proc. SPIE, Wavelet Applications in Signal and Image Processing VII,
%     Denver, CO, USA, 19-23 July, 1999, vol. 3813, pp. 422-431. 
% 	[3] T. Blu and M. Unser, "The fractional spline wavelet transform: definition and 
%	implementation," Proc. IEEE International Conference on Acoustics, Speech, and 
%	Signal Processing (ICASSP'2000), Istanbul, Turkey, 5-9 June 2000, vol. I, pp. 512-515 .

u=alpha/2-tau;
v=alpha/2+tau;
if real(alpha)<=-0.5
	disp('The autocorrelation of the fractional splines exists only ')
	disp('for degrees strictly larger than -0.5!')
	FFTanalysisfilters=[];
	FFTsynthesisfilters=[];
	return
end
nu=0:1/M:(1-1/M);

A=fractsplineautocorr(alpha,tau,nu);
A2=[A A];
A2=A2(1:2:length(A2));		% A2(z) = A(z^2)

if type(1)=='o'|type(1)=='O'
	% orthonormal spline filters
	lowa=sqrt(2)*((1+exp(2*i*pi*nu))/2).^(u+0.5).*((1+exp(-2*i*pi*nu))/2).^(v+0.5).*sqrt(A./A2);
	lowa(1+M/2)=0;
	lows=lowa;
	FFTanalysisfilters=[lowa];
	FFTsynthesisfilters=[lows];
else
	% semi-orthonormal spline filters
	lowa=sqrt(2)*((1+exp(2*i*pi*nu))/2).^(u+0.5).*((1+exp(-2*i*pi*nu))/2).^(v+0.5);
	lowa(1+M/2)=0;
	lows=lowa.*A./A2;
	
	if type(1)=='d'|type(1)=='D' 
		% dual spline wavelets
		FFTanalysisfilters=[lowa];
		FFTsynthesisfilters=[lows];
	else
		% B-spline wavelets
		if type(1)=='b'|type(1)=='B'
			FFTsynthesisfilters=([lowa]);
			FFTanalysisfilters=([lows]);
		else
			error(['''' type '''' ' is an unknown filter type!'])
		end
	end	
end



function A=fractsplineautocorr(alpha,tau,nu) 

% FRACTSPLINEAUTOCORR Frequency domain computation of fractional spline 
% 	autocorrelation.  A=fractsplineautocorr(alpha,tau,nu) computes the 
% 	frequency response of the autocorrelation filter A(exp(2*i*Pi*nu)) 
% 	of a fractional spline of degree alpha and shift tau.  It calls the function
%	FRACTSPLINEINTERPOL.
% 
% 	See also FFTFRACTSPLINEFILTERS, FRACTSPLINEINTERPOL
% 	
% 	Author: Thierry Blu, October 1999 revised July 2000
% 	Biomedical Imaging Group, EPFL, Lausanne, Switzerland.  
% 	
% 	References:
% 	[1] M. Unser and T. Blu, "Fractional splines and wavelets," 
% 	SIAM Review, Vol. 42, No. 1, pp. 43--67, January 2000.
% 	[2] M. Unser and T. Blu, "Construction of fractional spline wavelet bases," 
% 	Proc. SPIE, Wavelet Applications in Signal and Image Processing VII,
%     Denver, CO, USA, 19-23 July, 1999, vol. 3813, pp. 422-431. 
% 	[3] T. Blu and M. Unser, "The fractional spline wavelet transform: definition and 
%	implementation," Proc. IEEE International Conference on Acoustics, Speech, and 
%	Signal Processing (ICASSP'2000), Istanbul, Turkey, 5-9 June 2000, vol. I, pp. 512-515 .
%
%	Demo at http://bigwww.epfl.ch/demo/jfractsplinewavelet/
	
N=100;			% number of terms of the summation for computing
				% the autocorrelation frequency response

if real(alpha)<=-0.5
	disp('The autocorrelation of the fractional splines exists only ')
	disp('for degrees whose real part is strictly larger than -0.5!')
	A=[];
	return
end

if nargin<=2
	nu=tau;
	tau=(alpha+1)/2;
end

A=fractsplineinterpol(2*real(alpha)+1,2*i*imag(tau),nu);



function I=fractsplineinterpol(alpha,tau,nu) 

% FRACTSPLINEINTERPOL Frequency domain computation of the fractional spline 
% 	interpolation filter.  I=fractsplineinterpol(alpha,tau,nu) computes the 
% 	frequency response of the interpolation filter I(exp(-2*i*Pi*nu)) 
% 	of a fractional spline of degree alpha and shift tau.  It uses an acceleration 
% 	technique which improves the convergence of the infinite sum by 4 orders.
% 
% 	See also FFTFRACTSPLINEFILTERS, FRACTSPLINEAUTOCORR
% 	
% 	Author: Thierry Blu, October 1999 revised July 2000
% 	Biomedical Imaging Group, EPFL, Lausanne, Switzerland.  
% 	
% 	References:
% 	[1] M. Unser and T. Blu, "Fractional splines and wavelets," 
% 	SIAM Review, Vol. 42, No. 1, pp. 43--67, January 2000.
% 	[2] M. Unser and T. Blu, "Construction of fractional spline wavelet bases," 
% 	Proc. SPIE, Wavelet Applications in Signal and Image Processing VII,
%     Denver, CO, USA, 19-23 July, 1999, vol. 3813, pp. 422-431. 
% 	[3] T. Blu and M. Unser, "The fractional spline wavelet transform: definition and 
%	implementation," Proc. IEEE International Conference on Acoustics, Speech, and 
%	Signal Processing (ICASSP'2000), Istanbul, Turkey, 5-9 June 2000, vol. I, pp. 512-515 .
%
%	Demo at http://bigwww.epfl.ch/demo/jfractsplinewavelet/
	
N=10;			% number of terms of the summation for computing
				% the frequency response of the autocorrelation 

nu=nu-floor(nu);
nu1=1-nu;

V=zeros(1,length(nu));
V=1/alpha;
V=V+(0.5+nu)/N;
V=V+1/12*(alpha+1)*(1+6*nu+6*nu.*nu)/N^2;
V=V+1/12*(alpha+1)*alpha*(nu+1).*nu.*(2*nu+1)/N^3;
V=V/N^alpha.*(sin(pi*nu)/pi).^(alpha+1);
for n=(-N+1):(-1)
	V=V+abs(sinc(nu+n)).^(alpha+1);
end
V=V.*exp(2*i*pi*nu1*tau);

U=zeros(1,length(nu));
U=1/alpha;
U=U+(0.5-nu)/N;
U=U+1/12*(alpha+1)*(1-6*nu+6*nu.*nu)/N^2;
U=U-1/12*(alpha+1)*alpha*(nu-1).*nu.*(2*nu-1)/N^3;
U=U/N^alpha.*(sin(pi*nu)/pi).^(alpha+1);
for n=0:(N-1)
	U=U+abs(sinc(nu+n)).^(alpha+1);
end
U=U.*exp(-2*i*pi*nu*tau);

I=U+V;

function c = multinomial(n,varargin)
% MULTINOMIAL Multinomial coefficients
%
%   MULTINOMIAL(N, K1, K2, ..., Km) where N and Ki are floating point
%   arrays of non-negative integers satisfying N = K1 + K2 + ... + Km, 
%   returns the multinomial  coefficient   N!/( K1!* K2! ... *Km!).
%
%   MULTINOMIAL(N, [K1 K2 ... Km]) when Ki's are all scalar, is the 
%   same as MULTINOMIAL(N, K1, K2, ..., Km) and runs faster.
%
%   Non-integer input arguments are pre-rounded by FLOOR function.
%
% EXAMPLES:
%    multinomial(8, 2, 6) returns  28 
%    binomial(8, 2) returns  28
% 
%    multinomial(8, 2, 3, 3)  returns  560
%    multinomial(8, [2, 3, 3])  returns  560
%
%    multinomial([8 10], 2, [6 8]) returns  [28  45]

% Mukhtar Ullah
% November 1, 2004
% mukhtar.ullah@informatik.uni-rostock.de

nIn = nargin;
error(nargchk(2, nIn, nIn))

if ~isreal(n) || ~isfloat(n) || any(n(:)<0)
    error('Inputs must be floating point arrays of non-negative reals')
end

arg2 = varargin; 
dim = 2;

if nIn < 3                         
    k = arg2{1}(:).'; 
    if isscalar(k)
        error('In case of two arguments, the 2nd cannot be scalar')
    end    
else
    [arg2{:},sizk] = sclrexpnd(arg2{:});
    if sizk == 1
        k = [arg2{:}];        
    else
        if ~isscalar(n) && ~isequal(sizk,size(n))
            error('Non-scalar arguments must have the same size')
        end
        dim = numel(sizk) + 1; 
        k = cat(dim,arg2{:});              
    end    
end

if ~isreal(k) || ~isfloat(k) || any(k(:)<0)
    error('Inputs must be floating point arrays of non-negative reals')
end

n = floor(n);
k = floor(k);

if any(sum(k,dim)~=n)
    error('Inputs must satisfy N = K1 + K2 ... + Km ')
end

c = floor(exp(gammaln(n+1) - sum(gammaln(k+1),dim)) + .5); 

function [varargout] = sclrexpnd(varargin)
% SCLREXPND expands scalars to the size of non-scalars.
%    [X1,X2,...,Xn] = SCLREXPND(X1,X2,...,Xn) expands the scalar 
%    arguments, if any, to the (common) size of the non-scalar arguments,
%    if any.
%
%    [X1,X2,...,Xn,SIZ] = SCLREXPND(X1,X2,...,Xn) also returns the 
%    resulting common size in SIZ.  
%
% Example:
% >> c1 = 1; c2 = rand(2,3); c3 = 5; c4 = rand(2,3);
% >> [c1,c2,c3,c4,sz] = sclrexpnd(c1,c2,c3,c4)
%
% c1 =
%      1     1     1
%      1     1     1
% 
% c2 =
%     0.7036    0.1146    0.3654
%     0.4850    0.6649    0.1400
% 
% c3 =
%      5     5     5
%      5     5     5
% 
% c4 =
%     0.5668    0.6739    0.9616
%     0.8230    0.9994    0.0589
% 
% sz =
%      2     3

% Mukhtar Ullah
% November 2, 2004
% mukhtar.ullah@informatik.uni-rostock.de

C = varargin;
issC = cellfun('prodofsize',C) == 1;
if issC
    sz = [1 1];
else
    nsC = C(~issC);   
    if ~isscalar(nsC)
        for i = 1:numel(nsC), nsC{i}(:) = 0;  end
        if ~isequal(nsC{:})
            error('non-scalar arguments must have the same size')
        end
    end    
    s = find(issC);
    sz = size(nsC{1});      
    for i = 1:numel(s)
        C{s(i)} = C{s(i)}(ones(sz));
    end        
end
varargout = [C {sz}];

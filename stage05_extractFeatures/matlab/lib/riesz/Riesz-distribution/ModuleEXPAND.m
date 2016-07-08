function C=ModuleEXPAND(Cn,Hd);

Cn=fft2(Cn);
Cn=[Cn Cn; Cn Cn];
Cn=Cn.*repmat(Hd,[1 1 size(Cn,3)]);
C=ifft2(Cn); 

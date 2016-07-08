function Cn=ModuleREDUCE(C,H);

CF=fft2(C);
CF=CF.*repmat(H,[1 1 size(CF,3)]);
for iter=1:size(CF,3),
    CF(:,:,iter)=(CF(:,:,iter)+fftshift(CF(:,:,iter))+fftshift(CF(:,:,iter),1)+fftshift(CF(:,:,iter),2))/4;
end;
Cn=ifft2(CF(1:end/2,1:end/2,:)); 
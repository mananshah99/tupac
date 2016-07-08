function A = RieszVisualizeMultiscale(w,numberOfScales,order)

    N=33;
    Nmax=(N*2)-1;

    for i=1:numberOfScales-1,
        Nmax=Nmax*2-1;
    end;
    components=zeros(Nmax,Nmax,numberOfScales);
    countScale=1;
    
    for iterScale=1:order+1:numberOfScales*(order+1),
             
        wS=w(iterScale:iterScale+order);
        [I]=RieszViewTemplateN(wS', order, N, -1, '');
        margin=(Nmax-N)/2;
        
        I=I*sum(abs(wS))/mean(mean(abs(I))); % normalize each scale
                
        components(margin:end-margin-1,margin:end-margin-1,countScale)=I;
        N=(N*2)-1;
        
        countScale=countScale+1;
    end;

    A=0;
    for iterScale=1:numberOfScales,
        A=A+components(:,:,iterScale);
    end;
    
%     imagesc(A(270:510,270:510));set(gca,'xtick',[],'ytick',[]);axis image;
    figure;imagesc(A);set(gca,'xtick',[],'ytick',[]);axis image;

end


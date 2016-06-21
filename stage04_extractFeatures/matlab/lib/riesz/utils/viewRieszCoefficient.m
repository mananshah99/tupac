function viewRieszCoefficient(riesz)
for iterScale=1:size(riesz,2)-2,
    for iterRiesz=1:size(riesz{iterScale},3),
        figure;imshow(riesz{iterScale}(:,:,iterRiesz),[]);
    end;
end;

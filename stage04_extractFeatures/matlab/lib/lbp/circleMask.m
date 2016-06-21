function circle=circleMask(maskX, maskY, centerX, centerY, diameter)

    diameter = diameter + 2;

    [X, Y] = meshgrid(1:maskX, 1:maskY);
    
    circle = (Y - centerX).^2 + (X - centerY).^2 - (diameter/2).^2;
    circle = abs(circle) <= (diameter/2);
    
end

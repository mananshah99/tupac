function res = lesionLbpStat(mask, radius)

    [X, Y] = size(mask);
    res = [];
    
    for r=radius
        
        lbp = zeros(Y - 2*r, X - 2*r);

        for i=1+r:X - r
            for j=1+r:Y - r            
                center=mask(i,j);
                circle = circleMask(X, Y, i, j, r);
                circle_val = mask(1 == circle) > center;
                circle_power = linspace(size(circle_val,1),1,size(circle_val,1))';
                lbp(i,j) = sum(circle_val.^circle_power);
            end
        end

        % vectorize
        vect = lbp(lbp(:) > 0);
        
        res = [res ...
            min(vect),...
            max(vect),...
            median(vect),...
            mean(vect),...
            std(vect),...
            skewness(vect),...
            kurtosis(vect),...
            entropy(vect)...
        ];
    
    end
end
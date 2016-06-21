function res = lesionHistogram(image, mask, lower_thresh, upper_thresh, lenght)

    % vectorize
    vect = image(1 == mask(:));

    % counts
    range = linspace(lower_thresh,upper_thresh,lenght);
    res = hist(vect,range);
    
end
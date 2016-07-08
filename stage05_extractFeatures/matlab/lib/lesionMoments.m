function res = lesionMoments(image, mask)
   
    % vectorize
    vect = image(1 == mask(:));

    res = feature_vec(vect)';
    
end
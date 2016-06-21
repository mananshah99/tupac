function [image_box,mask_box] = lesionBox(image, mask)


    % init
    x_up=0; x_low=1000; y_up=0; y_low=1000;

    for i=1:size(mask,1)
        for j=1:size(mask,2)
            if mask(i,j) ~= 0
                if x_low > i
                    x_low=i;
                end
                if y_low > j
                    y_low=j;
                end
                if x_up < i
                    x_up=i;
                end
                if y_up < j
                    y_up=j;
                end
            end
        end
    end

    % segment
    image_box = image(x_low:x_up,y_low:y_up);
    mask_box = mask(x_low:x_up,y_low:y_up);
    
end
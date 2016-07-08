%% init
img = rand(100);
mask = img > 0.3;

%% lesion
% a lesion is defined as an area of an image in a mask
% lesion = img .* mask

% croped lesion
[img_box,mask_box] = lesionBox(img, mask);

% histogram of the lesion
lesion_histogram = lesionHistogram(img, mask, 0, 1, 40);

%% features 
features = [];
features = [features lesionStat(img, mask)]; % lesion statistical features
features = [features lesion_histogram]; % lesion histogram
features = [features histogramStats(lesion_histogram)];
features = [features lesionRieszEnergies(img, mask, 2, 5)]; % riesz features
features = [features lesionMoments(img, mask)];
features = [features lesionLbpStat(mask, 3:5)];
features = [features lesionCoocurenceStat(img_box, mask_box, 0, 1, 10, [0 1; -1 1])];
features = [features lesionGaborStat(img_box, mask_box, 5, 8, 39, 39, 4, 4)];
features = [features lesionWaveletStat(img_box, mask_box, 'db2', 4)];

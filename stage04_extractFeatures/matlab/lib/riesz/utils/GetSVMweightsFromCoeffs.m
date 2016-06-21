function w=GetSVMweightsFromCoeffs(features1,features2)

% SVM parameters for libsvm
svm_type=0;    %  C-SVC
kernel_type=2; % 2: Gaussian kernel
gamma=10^(0);  % 'width' of the Gaussian kernel
cost=10^2;     % C parameter in loss function

options = ['-s ', num2str(svm_type),...
' -t ', num2str(kernel_type),...
' -g ', num2str(gamma),...
' -c ', num2str(cost)];

classesTrain=[ones(size(features1,1),1);-ones(size(features2,1),1)];
featuresTrain=[features1;features2];

model=svmtrain(classesTrain,featuresTrain,options);

w = model.SVs'*model.sv_coef; % primal-dual relationship
if model.Label(1) == -1,
  w = -w;
end;
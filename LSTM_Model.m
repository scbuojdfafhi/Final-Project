addpath(genpath(pwd));
% load training set and testing set
load('ECG_train.mat')
load('ECG_test.mat')

X_train=cast(X_train,'double');
% define function of wavelet transform which will be used to create feature map, where
% '5422' is length of signal data, '128' is sampling frequency

sf = waveletScattering('Signallength',5422,'SamplingFrequency',128);
% Utilize sf to create feature matrix to see the size of matrix (demo)
feat = featureMatrix(sf,X_train(1,:)); % size :203*11

% Applied wavelet transform into all the signal data in training set and
% testing set to create feature matrix.
scat_feat_train = cell(72,1);
scat_feat_test = cell(30,1);
X_train = cast(X_train,'double');
X_test = cast(X_test,'double');
parfor itr = 1:72
    tmp = featureMatrix(sf,X_train(itr,:));
    scat_feat_train{itr,1}=tmp;
end

parfor ttr = 1:30
    tmp = featureMatrix(sf,X_test(ttr,:));
    scat_feat_test{ttr,1}=tmp;
end

% Train LSTM model in training set

% define inputsize
[inputSize,~] = size(feat);

% set basic parameter of training model
numHiddenUnits = 500;
numClasses = 2;
maxEpoches = 200;
minBatchSize=100;

% change type of label of training set as category
Y_train = categorical(y_train');

% create LSTM model
layers = [...
    sequenceInputLayer(inputSize)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
% set various parameters of LSTM model
options = trainingOptions( 'adam',...
    'InitialLearnRate',0.0001,...
    'LearnRateSchedule','none',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',50,...
    'MaxEpochs',maxEpoches,...
    'MiniBatchSize',minBatchSize,...
    'SequenceLength','shortest',...
    'Shuffle','every-epoch',...
    'Plots','training-progress');

% train model
netscat = trainNetwork(scat_feat_train,Y_train,layers,options);

% Evaluate LSTM model in testing set

% change type of label of testing set into category
Ytest = categorical(y_test');
% predict testing set, and show prediction results of each instance
ypred = classify(netscat,scat_feat_test,'MiniBatchSize',minBatchSize,'SequenceLength','shortest');
% calculate accuracy
accuracy = round((sum(ypred==Ytest)./numel(Ytest))*100);
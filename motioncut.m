function cutmatrix = motioncut(filename, time, len,h,w, para)
addpath(genpath('gco-v3.0'));

file = ['interval-',filename,'-',time,'us','.mat'];
intervals = load(file);
% len = 100;
% para = 0.5;
% intervals = load('interval-train-350kmh-2500us.mat');
h = double(h);
w = double(w);
len = fix(len);
len1 = len;
if len1 > 500 %800
    len1 = 500;
end
probmatrix = zeros(h,w,len);
for i = 1 : h
    for j = 1 : w
        seq = intervals.itv{(i-1)*w+j};
        seq = double(seq(:));
        %seq = 1./seq;
        itvsum = 0;
        for k = 1 : length(seq)
            itvsum = itvsum + seq(k);
            if itvsum > len1
                break
            end
        end
        trBG.m=mean(seq(2:k));
        trBG.c=std(seq(2:k))+mean(seq(2:k))*para;%+mean(seq)/4;
        clear proby1;
        for k=1:length(seq)
            maxproby = normpdf(trBG.m,trBG.m,trBG.c);
            proby =  normpdf(seq(k),trBG.m,trBG.c);
            proby1(k) = 1 - proby./maxproby;
        end
        flag = 1;
        for k=1:length(seq)
            for l = 1:intervals.itv{(i-1)*w+j}(k)
                probmatrix(i,j,flag) = proby1(k);
                flag = flag + 1;
            end
             if flag > len
                    break
             end
        end
    end
end
% 
% a = probmatrix(:,:,200);
% imshow(a)

opt.lambda = 5;
% call GCO to run graph cuts
hMRF = GCO_Create(prod([h,w]),2);
GCO_SetSmoothCost( hMRF, [0 1;1 0] );
AdjMatrix = getAdj([h,w]);
amplify = 10 * opt.lambda;
GCO_SetNeighbors( hMRF, amplify * AdjMatrix );
OmegaOut = false([h*w, 1]);
beta = 0.5*(std(probmatrix(:)))^2;
gamma = opt.lambda * beta;
energy_cut = 0;
cutmatrix = zeros(h,w,len);
for i = 1 :  len
    E = probmatrix(:,:,i);
    E = E(:);

    GCO_SetDataCost(hMRF, (amplify/gamma)*[ 0.5*(E).^2, ~OmegaOut(:)*beta + OmegaOut(:)*0.5*max(E(:)).^2]' );
    GCO_Expansion(hMRF);
    Omega(:) = ( GCO_GetLabeling(hMRF) == 1 )';
    energy_cut = energy_cut + double( GCO_ComputeEnergy(hMRF) );

    ObjArea = sum(Omega(:)==0);
    energy_cut = (gamma/amplify) * energy_cut;

    %figure
    a = reshape(Omega,[h,w]);
    
    cutmatrix(:,:,i) = 1 -a;
    if sum(sum(cutmatrix(:,:,i))) > 0.7*h*w
        cutmatrix(:,:,i) = ones(h,w);
    end
    %imshow(1-a)
end
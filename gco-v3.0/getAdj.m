function W = getAdj(sizeData)
numSites = prod(sizeData);
id1 = [1:numSites, 1:numSites, 1:numSites];
id2 = [ 1+1:numSites+1,...
        1+sizeData(1):numSites+sizeData(1),...
        1+sizeData(1)*sizeData(2):numSites+sizeData(1)*sizeData(2)];
value = ones(1,3*numSites);
W = sparse(id1,id2,value);
W = W(1:numSites,1:numSites);
end
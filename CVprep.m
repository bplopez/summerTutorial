% Prepares CV data sets

%% Format data
load('bReconImage')
images = uint8(zeros(96,64,10,2));
for ii = 1:10
    for jj = 1:2
        images(:,:,ii,jj) = uint8(reconImage{ii,jj});
    end
end
clear ii jj

%% 9 min vs 1 min
cv91 = uint8(zeros(96,64,10,2,2)); %row, col, comb, detector, train v. test

train = nchoosek(1:10,9);
test = flipud(nchoosek(1:10,1));

for ii = 1:10
    for jj = 1:2
        cv91(:,:,ii,jj,1) = sum(images(:,:,train(ii,:),jj),3);
        cv91(:,:,ii,jj,2) = sum(images(:,:,test(ii,:),jj),3);
    end
end
clear ii jj train test

%% 5 min vs 5 min
cv55 = uint8(zeros(96,64,2,2,2));

train = [1 2 3 4 5;6 7 8 9 10];
test = [6 7 8 9 10; 1 2 3 4 5];

for ii = 1:2
    for jj = 1:2
        cv55(:,:,ii,jj,1) = sum(images(:,:,train(ii,:),jj),3);
        cv55(:,:,ii,jj,2) = sum(images(:,:,test(ii,:),jj),3);
    end
end
clear ii jj train test

%% Det 1 vs Det 2
cv12 = uint8(zeros(96,64,2,2)); % row, col, comb, train v test

cv12(:,:,1,1) = sum(images(:,:,:,1),3);
cv12(:,:,1,2) = sum(images(:,:,:,2),3);
cv12(:,:,2,1) = sum(images(:,:,:,2),3);
cv12(:,:,2,2) = sum(images(:,:,:,1),3);

save('bReconImage')

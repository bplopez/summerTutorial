% CVdriver

clear all
load('imagesRaw','imRaw');
ind = 1:3;
b = sum(imRaw(:,:,ind,1),3);
cv = sum(imRaw(:,:,7:9,1),3);
[M,N] = size(b);
MN = M*N;

%% Differential operator
phix = zeros(MN,MN);
phiy = zeros(MN,MN);
for ii = 1:MN-M
    phix(ii,ii) = -1;
    phix(ii,ii+M) = 1;
end
phix(ii+1:end,:) = phix(ii-M+1:ii,:);
for ii = 1:MN
    if rem(ii,M) ~= 0
        phiy(ii,ii) = -1;
        phiy(ii,ii+1) = 1;
    else
        phiy(ii,ii-1) = -1;
        phiy(ii,ii) = 1;
    end
end
phi = [phix;phiy];

clear M N MN phix phiy ii ind;

%% Algorithm

%mu = 0.1;
mu = [1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1]';
tau = mu*2;
iter = 50;

l = length(mu);
iters = zeros(l,1);
us = zeros(96,64,l);
resCV = zeros(l,1);

%[u,resP,resD,resC,resb,resCV]= ADMMsolve_CV(b,cv,mu,tau,phi,iter);
%[u,resP,resD,resb,resT,resC] = ADMMsolve(b,mu,tau,phi,iter);
for ii = 1:length(mu)
    u = ADMMsolve(b,mu(ii),tau(ii),phi,iter);
    iters(ii) = size(u,3);
    us(:,:,ii) = u(:,:,end);
    resCV(ii) = norm(cv-us(:,:,ii));
    fprintf('Finished\n')
end

clear phi imRaw iter mu tau
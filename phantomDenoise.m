% Run solvers on phantom

%{
max_iter = 50;
%mu = [0.01 0.1 1 10 100];
%tau = mu;
mu = [1 10 100];
tau = [0.01 0.1 1 10 100];
Nmu = length(mu);
Ntau = length(tau);

im = phantom(64);
b = imnoise(im,'gaussian');
[M,N] = size(b);
MN = M*N;
xin = rand(M,N);

xout = zeros(Nmu,Ntau,M,N,max_iter+1);
wout = zeros(Nmu,Ntau,2*M,N,max_iter+1);
lout = wout;
pxout = wout;
n1 = zeros(Nmu,Ntau,max_iter+1);
n2 = n1;

for ii=1:Nmu
for jj = 1:Ntau
[xout(ii,jj,:,:,:),wout(ii,jj,:,:,:),lout(ii,jj,:,:,:),pxout(ii,jj,:,:,:),n1(ii,jj,:),n2(ii,jj,:)] = L1solve(b,xin,mu(ii),tau(jj),max_iter);
fprintf('%2.2f %2.2f\n',mu(ii),tau(jj));
end
end

%[xout,wout,lout,n1,n2] = L1solve(b,xin,mu,tau,max_iter);
%}

im = phantom(64);
b = imnoise(im,'gaussian');

[M,N] = size(b);
MN = M*N;
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
phi = phix+phiy;
%phi = [phix;phiy];
clear M N MN phix phiy;

mu = 0.1;
% tau = 0.01;	% AMA
tau = 0.15;
%tau = (mu/20)^(1/3);	% ADMM
iter = 25;

%[x, px, l, lh, w, a, n, r] = AMAsolve(b,mu,tau,phi,iter);
[x, px, l, lh, w, wh, a, c, n, r] = ADMMsolve(b,mu,tau,phi,iter);

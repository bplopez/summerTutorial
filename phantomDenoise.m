% Run L1solve on phantom

%{
max_iter = 20;
mu = 1;
tau = 0.1;
%}
max_iter = 20;
mu = [0.01 0.1 1 10 100];
tau = mu;
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
end
end

%[xout,wout,lout,n1,n2] = L1solve(b,xin,mu,tau,max_iter);

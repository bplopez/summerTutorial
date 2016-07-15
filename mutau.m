% See how mu and tau change the resulting image

max_iter = 20;
mu = 1;
tau = 1;
%mu = [100 10 1 0.1 0.01];
%tau = [100 10 1 0.1 0.01];
im = load('p0105a_TT_1-300000_LL_all_EE_126.45-154.55.mat');
b = im.imageDet1;
[M,N] = size(b);
MN = M*N;
xin = rand(M,N);

%{
for ii = 1:length(tau)
[~,~,~,~,t2(:,ii)] = L1solve(b,xin,mu,tau(ii),max_iter);
end
%}

[xout,wout,lout,pxout,n1,n2] = L1solve(b,xin,mu,tau,max_iter);

%{
xout = zeros(length(mu),length(tau),MN);

for ii = 1:length(mu)
 for jj = 1:length(tau)
  xout(ii,jj,:) = L1solve(b,xin,mu(ii),tau(jj),max_iter);
end
end

%}
%{
for ii = 1:5
 for jj = 1:5
  figure
  plotImage(reshape(xout(ii,jj,:),M,N));
 end
end
%}

% Run solvers on phantom

s = 1;  % 1 = ADMM, 2 = AMA

%% Load image
im = phantom(64);
gsl = 256;
m = 0;
v = 0.01;
b = imnoise(im,'gaussian',m,v/gsl);
b = round(b*256);
%b = round(im*256);
[M,N] = size(b);
MN = M*N;

clear m v gsl

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
clear M N MN phix phiy ii;

%% Run solver
mu = 0.1;
iter = 50;

if s == 1   % ADMM
    tau = mu*2;   
    [x, px, l, lh, w, wh, a, c, n, r] = ADMMsolve(b,mu,tau,phi,iter);
else
    tau = mu/10;	% AMA
    [x, px, l, lh, w, a, n, r] = AMAsolve(b,mu,tau,phi,iter);
end

clear phi

%% Plot results
ii = length(r);
figure(1); 
subplot(2,3,1); imagesc(round(x(:,:,ii))); colorbar; axis off; 
    title(sprintf('\nx')); 
subplot(2,3,4); imagesc(px(:,:,ii)); colorbar; title('px'); axis off;
if s == 1
    subplot(2,3,2); imagesc(w(:,:,ii+1)); colorbar; 
        title(sprintf('iteration = %d\nw',ii)); axis off;
    subplot(2,3,5); imagesc(wh(:,:,ii+1)); colorbar; title('wh'); axis off;
else
    subplot(2,3,2); imagesc(w(:,:,ii)); colorbar; axis off; 
        title(sprintf('iteration = %d\nw',ii)); 
end
subplot(2,3,3); imagesc(l(:,:,ii+1)); colorbar; title('l'); axis off;
subplot(2,3,6); imagesc(lh(:,:,ii+1)); colorbar; title('lh'); axis off;

jj = 32;
figure(2); plot(b(:,jj)); hold on; plot(x(:,jj,end)); hold off; 
    legend('Input','Output','location','best');
    title(sprintf('Column %d',jj));

figure(3); 
subplot(1,2,1); imagesc(b); colorbar; axis off; title('Noisy Image')
subplot(1,2,2); imagesc(round(x(:,:,end))); colorbar; axis off; title('TV Image')

clear ii jj s;

%% old mu vs. tau calculations
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
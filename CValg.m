% Cross validation algorithm

%% tau analysis
%
tic
load('bReconImage','cv12');
cd a12mt
comb = 2;
data = cv12(:,:,1:comb,:);
M = size(data,1);
N = size(data,2);
MN = M*N;

% Differential operator
phix = (zeros(MN,MN));
phiy = (zeros(MN,MN));
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

% Run solver

mu = [1 0.5 0.1 0.05 0.01]; 
tau = [10 1 0.5 0.2 0.1 0.05 0.02 0.01 0.001];
Nmu = length(mu);
Ntau = length(tau);

iter = 50;
x = zeros(96,64,Nmu,Ntau,comb);
pres = zeros(Nmu,Ntau,comb);
dres = zeros(Nmu,Ntau,comb);
iters = zeros(Nmu,Ntau,comb);
cvres = zeros(Nmu,Ntau,comb);

x_cell = cell(Nmu,Ntau,comb);
pres_cell = cell(Nmu,Ntau,comb);
dres_cell = cell(Nmu,Ntau,comb);

for cc = 1:comb
for mm = 1:Nmu
for tt = 1:Ntau
    fprintf('\nStarting comb = %d, mu = %2.3f, tau = %2.3f',cc,mu(mm),tau(tt));
	[x_cell{mm,tt,cc},pres_cell{mm,tt,cc},dres_cell{mm,tt,cc}] = ...
        ADMMsolve(double(data(:,:,cc,1)),mu(mm),tau(tt),phi,iter);
	x(:,:,mm,tt,cc) = x_cell{mm,tt,cc}(:,:,end);
	pres(mm,tt,cc) = pres_cell{mm,tt,cc}(end);
	dres(mm,tt,cc) = dres_cell{mm,tt,cc}(end);
	iters(mm,tt,cc) = length(pres_cell{mm,tt,cc});
	fprintf('\nIterations = %d\n',iters(mm,tt,cc));
	cvres(mm,tt,cc) = norm(x(:,:,mm,tt,cc)-double(data(:,:,cc,2)))^2;
end
end
end

clear iter phi cc mm tt
save('b12mt')
toc
%}

%% mu analysis
%{
load('aReconImage','cv91');
comb = 1;
det = 1;
data = cv91(:,:,1:comb,1:det,:);
M = size(data,1);
N = size(data,2);
MN = M*N;

% Differential operator
phix = (zeros(MN,MN));
phiy = (zeros(MN,MN));
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

% Run solver

%mu = [0.5 0.1 0.05 0.01 0.005];
%Nmu = length(mu);
%tau = mu*2;
%tau = 0.2;

tau = [0.3 0.2 0.1 0.05 0.03];
Nmu = length(tau);
mu = ones(1,Nmu)*0.1;

iter = 50;
x = zeros(96,64,Nmu,comb,det);
pres = zeros(Nmu,comb,det);
dres = zeros(Nmu,comb,det);
iters = zeros(Nmu,comb,det);
cvres = zeros(Nmu,comb,det);
for ii = 1:comb
for jj = 1:det
for kk = 1:Nmu
	%fprintf('\nStarting comb = %d, det = %d, mu = %2.3f',ii,jj,mu(kk));
    fprintf('\nStarting comb = %d, det = %d, tau = %2.3f',ii,jj,tau(kk));
	[x_temp,pres_temp,dres_temp] = ADMMsolve(double(data(:,:,ii,jj,1)),mu(kk),tau(kk),phi,iter);
	x(:,:,kk,ii,jj) = x_temp(:,:,end);
	pres(kk,ii,jj) = pres_temp(end);
	dres(kk,ii,jj) = dres_temp(end);
	iters(kk,ii,jj) = length(pres_temp);
	fprintf('\nIterations = %d\n',iters(kk,ii,jj));
	cvres(kk,ii,jj) = norm(x(:,:,kk,ii,jj)-double(data(:,:,ii,jj,2)))^2;
end
end
end
clear iter phi ii x_temp pres_temp dres_temp
%save('a55')
%}

%% Plot results
%{
figure(1); 
subplot(1,3,1); plotImage(data(:,:,1)); colorbar; title('Train');
subplot(1,3,2); plotImage(data(:,:,2)); colorbar; title('Test');
subplot(1,3,3); plotImage(x(:,:,end)); colorbar; title('Result');

figure(2);
subplot(1,2,1); semilogy(pres); title('Primal Residual');
subplot(1,2,2); semilogy(dres); title('Dual Residual');

jj = 42;
figure(7); 
plot(data(:,jj,1)); hold on; plot(x(:,jj,end)); hold off; 
legend('Input','Output','location','best');
title(sprintf('Column %d',jj));
clear jj
%}

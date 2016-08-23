% TVdriver

%zz=4;

%% Load image
%im = load('p0105a_TT_1-300000_LL_all_EE_126.45-154.55.mat');
%b = im.imageDet1*zz;
%im = imnoise(phantom(64),'gaussian');
%b = round(im*256);
b = double(imSum(:,:,10,1));
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

clear M N MN phix phiy ii;

%% Run solver
s = 1;  % 1 = ADMM, 2 = AMA
mu = 0.1;
iter = 50;
if s == 1   % ADMM
    tau = mu*2;   
    %[x,pres,dres,r,n,c,a,px,l,lh,w,wh] = ADMMsolve(b,mu,tau,phi,iter);
    [x,pres,dres,r,n,c,a] = ADMMsolve(b,mu,tau,phi,iter);
    %[x,pres,dres,tau,as,bs] = AADMMsolve(b,mu,tau,phi,iter);
else
    tau = mu/10;	% AMA
    %[x,r,n,a,px,l,lh,w] = AMAsolve(b,mu,tau,phi,iter);
    [x,r,n,a] = AMAsolve(b,mu,tau,phi,iter);
end

clear phi

%% Plot results
%{
ii = length(r);
figure(1); 
subplot(2,3,1); imagesc(round(x(:,:,ii+1))); colorbar; axis off; 
    title(sprintf('\nx')); 
subplot(2,3,4); imagesc(px(:,:,ii+1)); colorbar; title('px'); axis off;
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
%}
%{
jj = 42;
figure(3); 
%subplot(2,2,zz)
plot(b(:,jj)); hold on; plot(x(:,jj,end)); hold off; 
legend('Input','Output','location','best');
title(sprintf('Column %d',jj));
%
figure(4); 
subplot(1,2,1); plotImage(b); colorbar; title('Noisy Image')
subplot(1,2,2); plotImage(round(x(:,:,end))); colorbar; title('TV Image')
%}
%{
figure(4);
subplot(2,2,zz); plotImage(round(x(:,:,end))); colorbar; title('TV Image')
%}
clear ii jj s
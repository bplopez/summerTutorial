% TVdriver

%% Load image
im = load('p0105a_TT_1-300000_LL_all_EE_126.45-154.55.mat');
b = im.imageDet1;
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
mu = 1;
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

jj = 42;
figure(2); plot(b(:,jj)); hold on; plot(x(:,jj,end)); hold off; 
    legend('Input','Output','location','best');
    title(sprintf('Column %d',jj));

figure(3); 
subplot(1,2,1); plotImage(b); colorbar; title('Noisy Image')
subplot(1,2,2); plotImage(round(x(:,:,end))); colorbar; title('TV Image')

clear ii jj s
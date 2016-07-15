% Plot resulting final images

iter = max_iter+1;

%% Make movies
movX(iter) = struct('cdata',[],'colormap',[]);
movW = movX;
movPX = movX;
movN1 = movX;
movN2 = movX;

for kk = 1:iter
    for ii = 1:5
        for jj = 1:5
            figure(1)
            subplot(5,5,5*(ii-1)+jj,'replace');
            imagesc(squeeze(xout(ii,jj,:,:,kk)));
            axis equal; axis off;
            figure(2)
            subplot(5,5,5*(ii-1)+jj,'replace');
            imagesc(squeeze(pxout(ii,jj,:,:,kk)));
            axis off;
            figure(3)
            subplot(5,5,5*(ii-1)+jj,'replace');
            imagesc(squeeze(wout(ii,jj,:,:,kk)));
            axis off
        end
    end
    figure(4); surf(n1(:,:,kk)); view(110,30);
    figure(5); surf(n2(:,:,kk)); view(110,30);
    movX(kk) = getframe(1);
    movPX(kk) = getframe(2);
    movW(kk) = getframe(3);
    movN1(kk) = getframe(4);
    movN2(kk) = getframe(5);
end

%% Display movies
%{
figure; axes('pos',[0 0 1 1],'visible','off'); movie(movX,1,1)
figure; axes('pos',[0 0 1 1],'visible','off'); movie(movPX,1,1)
figure; axes('pos',[0 0 1 1],'visible','off'); movie(movW,1,1)
figure; axes('pos',[0 0 1 1],'visible','off'); movie(movN1,1,1)
figure; axes('pos',[0 0 1 1],'visible','off'); movie(movN2,1,1)
%}

%% Calculate true residual
%{
temp = phi*trueIM(:);
truePX = [reshape(temp(1:MN,:),M,N);reshape(temp(MN+1:2*MN,:),M,N)];
matN1true = zeros(size(matN1));
matN2true = zeros(size(matN2));
for kk = 1:21
    for ii = 1:5
        for jj = 1:5
            matN1true(ii,jj,kk) = norm(truePx - squeeze(matW(ii,jj,:,:,kk)))/norm(truePX);
            matN2true(ii,jj,kk) = norm(squeeze(matX(ii,jj,:,:,kk))-im)/norm(im);
        end
    end
end
%}

%% Plot residuals
%{
for ii = 1:5
for jj = 1:5
figure(4)
subplot(5,5,5*(ii-1)+jj,'replace');
plot(squeeze(matN1(ii,jj,:))); xlim([0 21]);
figure(5)
subplot(5,5,5*(ii-1)+jj,'replace');
plot(squeeze(matN1true(ii,jj,:))); xlim([0 21]);
figure(6)
subplot(5,5,5*(ii-1)+jj,'replace'); 
plot(squeeze(matN2(ii,jj,:))); xlim([0 21]);
figure(7)
subplot(5,5,5*(ii-1)+jj,'replace');
plot(squeeze(matN2true(ii,jj,:))); xlim([0 21]);
end
end
%}
%% Create gradient
%{
phix = zeros(MN,MN);
phiy = zeros(MN,MN);
for ii = 1:MN-M
    phix(ii,ii) = -1;
    phix(ii,ii+M) = 1;
end
for ii = 1:MN
    if rem(ii,M) ~= 0
        phiy(ii,ii) = -1;
        phiy(ii,ii+1) = 1;
    end
end
phi = [phix;phiy];
%}


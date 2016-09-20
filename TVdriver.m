% TVdriver: Applies either the Fast ADMM solve (ADMMsolve.m) or the Fast
% AMA solve (AMAsolve.m) to MATLAB's phantom [M x N] image with Gaussian noise.
% This function generates the necessary forward finite difference operator
% for the image.
%
% Call:     x = TVdriver(solver, mu, iter)
%
% Inputs:   solver = 1 (for ADMMsolve.m) and 2 (for AMAsolve.m)
%           mu = strongly complex constant (typically 0.1 - 0.01)
%           iter = maximum number of iterations in algorithm
%
% Outputs:  x = resulting denoised images iterations [ M x N x iter ]

function x = TVdriver(solver, mu, iter)

%% Load image
im = imnoise(phantom(64),'gaussian');
b = round(im*256);
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
%solver = 1;  % 1 = ADMM, 2 = AMA
%mu = 0.1;
%iter = 50;
if solver == 1   % ADMM
    tau = mu*2;   
    x = ADMMsolve(b,mu,tau,phi,iter);
else
    tau = mu/10;	% AMA
    x = AMAsolve(b,mu,tau,phi,iter);
end

clear phi

end
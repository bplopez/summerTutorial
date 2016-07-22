% AMAsolve for min |w| + (mu/2) ||Ax-b||^2 s.t. phi*x = w
%
% Inputs:     b = measured image [ M x N ]
%             mu = strongly complex constant
%             tau = step size (< mu/8)
%             max_iter = maximum number of iterations
%         
% Outputs:    x_out = image iterations [ M x N x iter ]
%             n = norm of image iteration differences
            

function  [x,px,l, lh, w, a, n, r] = AMAsolve(b,mu,tau,phi,max_iter)

%% Set up variables

% Image space
[M,N] = size(b);
MN = M*N;
b = b(:);

% Gradient operator
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
phix = phi(1:MN,:); phiy = phi(MN+1:2*MN,:);

% Physics model
%A = eye(MN,MN);

%% AMA algorithm
%
% Step size restriction
if tau >= mu/8
    tau = mu/16;
end

% Stopping criteria
e = 0.005;

% Iteration variables initialization
x_mat = zeros(MN,max_iter);
px_mat = zeros(2*MN,max_iter);
w_mat = zeros(2*MN,max_iter);
l_mat = zeros(2*MN,max_iter+1);
lh_mat = zeros(2*MN,max_iter+1);
a = zeros(max_iter+1,1); a(1) = 1;
n = zeros(max_iter,1);
r = zeros(max_iter,1);

for k = 1:max_iter
        
    % L2 solve
    x_mat(:,k) = b - (phix*lh_mat(1:MN,k) + phiy*lh_mat(MN+1:2*MN,k))/mu;
    %x_mat(:,k) = b - (phi'*lh_mat(:,k))/mu;
    
    % L1 solve
    px_mat(:,k) = phi*x_mat(:,k);
    w_soft = px_mat(:,k) + lh_mat(:,k)*mu;
    w_mat(:,k) = wthresh(w_soft,'s',(1/tau));
    
    % Lagrange update with acceleration
    l_mat(:,k+1) = lh_mat(:,k) + tau*(w_mat(:,k) - px_mat(:,k));
    a(k+1) = (1+sqrt(1+4*a(k)^2))/2;
    lh_mat(:,k+1) = l_mat(:,k+1) + (a(k) -1)/(a(k+1))*(l_mat(:,k+1) - l_mat(:,k));
    
    r(k) = norm(x_mat(:,k)-b);
    
    % Stopping criterion
    if k == 1
        n(k) = 1; 
        continue
    else
        n(k) = norm(x_mat(:,k)-x_mat(:,k-1))/norm(x_mat(:,k));
        if n(k) < e
            break
        end
    end
    
end

%% Clean up variables
x = reshape(x_mat(:,1:k),M,N,k);
px = [reshape(px_mat(1:MN,1:k),M,N,k); ...
          reshape(px_mat(MN+1:2*MN,1:k),M,N,k)];
w = [reshape(w_mat(1:MN,1:k),M,N,k); ...
         reshape(w_mat(MN+1:2*MN,1:k),M,N,k)];
l = [reshape(l_mat(1:MN,1:k+1),M,N,k+1); ...
         reshape(l_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
lh = [reshape(lh_mat(1:MN,1:k+1),M,N,k+1); ...
          reshape(lh_mat(MN+1:2*MN,1:k+1),M,N,k+1)]; 
a = a(1:k+1);
n = n(1:k);
r = r(1:k);
%}

%% AMA
%{
% Step size restriction
if tau >= mu/8
    tau = mu/16;
end

% Stopping criteria
e = 0.005;

% Iteration variables initialization
x_mat = zeros(M,N,max_iter);
px_mat = zeros(M,N,2,max_iter);
w_mat = zeros(M,N,2,max_iter);
l_mat = zeros(M,N,2,max_iter+1);
lh_mat = zeros(M,N,2,max_iter+1);
a = zeros(max_iter+1,1); a(1) = 1;
n = zeros(max_iter,1);
r = zeros(max_iter,1);

for k = 1:max_iter
        
    % L2 solve
    [glx,~] = gradient(lh_mat(:,:,1,k));
    [~,gly] = gradient(lh_mat(:,:,2,k));
    x_mat(:,:,k) = b - (glx+gly)/mu;
    
    % L1 solve
    [px_mat(:,:,1,k), px_mat(:,:,2,k)] = gradient(x_mat(:,:,k));
    for ii=1:2
        w_soft = px_mat(:,:,ii,k) + lh_mat(:,:,ii,k)*tau;
        w_mat(:,:,ii,k) = wthresh(w_soft,'s',(1/tau));
    end
    
    % Lagrange update with acceleration
    a(k+1) = (1+sqrt(1+4*a(k)^2))/2;
    for ii = 1:2
    l_mat(:,:,ii,k+1) = lh_mat(:,:,ii,k) + tau*(w_mat(:,:,ii,k) - px_mat(:,:,ii,k));
    lh_mat(:,:,ii,k+1) = l_mat(:,:,ii,k+1) + (a(k) -1)/(a(k+1))*(l_mat(:,:,ii,k+1) - l_mat(:,:,ii,k));
    end
    
    % Residual
    r(k) = norm(x_mat(:,:,k)-b);
    
    % Stopping criterion
    if k == 1
        n(k) = 1; 
        continue
    else
        n(k) = norm(x_mat(:,:,k)-x_mat(:,:,k-1))/norm(x_mat(:,:,k));
        if n(k) < e
            break
        end
    end
    
end

%% Clean up variables
x = x_mat(:,:,1:k);
px = px_mat(:,:,:,1:k);
w = w_mat(:,:,:,1:k);
l = l_mat(:,:,:,1:k+1);
lh = lh_mat(:,:,:,1:k+1);
a = a(1:k+1);
n = n(1:k);
r = r(1:k);
%}
end

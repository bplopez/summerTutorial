% ADMMsolve: applies algorithm from Goldstein paper
%
% Inputs:     b = measured image [ M x N ]
%             mu = strongly complex constant
%             tau = step size (< mu/8)
%             phi = forward finite difference (2D directions) [ 2MN x MN ]
%             max_iter = maximum number of iterations
%         
% Outputs:    x = image iterations [ M x N x iter ]
%             px = phi*(image iterations) [ 2M x N x iter]
%             l = Lagrange multiplier [ 2M x N x (iter+1) ]
%             lh = Lagrange multiplier [ 2M x N x (iter+1) ]
%             w = gradient image iterations [ 2M x N x (iter+1) ] 
%             wh = gradient image iterations [ 2M x N x (iter+1) ]
%             a = acceleration factor [ (iter+1) x 1]
%             c = combined residual [ (iter+1) x 1 ]
%             n = norm of image iteration differences [ iter x 1 ]
%             r = norm of residual with input image [ iter x 1 ]

function [x,px,l,lh,w,wh,a,c,n,r] = ADMMsolve(b,mu,tau,phi,max_iter)

%
%% Set up variables

% Image size
[M,N] = size(b);
MN = M*N;
b = b(:);

% Gradient operator
%phi = phi(1:MN,:)+phi(MN+1:2*MN,:);
L2A = mu*eye(MN,MN) + tau*(phi'*phi);	% constant used in L2 subproblem

%% ADMM algorithm

% Step size restriction
if tau^3 > mu/16
tau = (mu/20)^(1/3);
end

% Stopping criteria
e = 0.005;

% Residual difference factor
eta = 0.999;

% Iteration variables
x_mat = zeros(MN,max_iter);
px_mat = zeros(2*MN,max_iter);
w_mat = zeros(2*MN,max_iter+1);
wh_mat = zeros(2*MN,max_iter+1);
w_mat(:,1) = wh_mat(:,1);
l_mat = zeros(2*MN,max_iter+1);
lh_mat = zeros(2*MN,max_iter+1);
l_mat(:,1) = lh_mat(:,1);
a = zeros(max_iter+1,1); a(1) = 1;	% acceleration factor
n = zeros(max_iter,1); 			% ||x-x*||/||x*|| for stopping
r = zeros(max_iter,1);			% residual ||x-b||
c = zeros(max_iter+1,1); c(1) = 10^6;	% combined residual

% Iteration
for k = 1:max_iter

    % L2 solve
    x_mat(:,k) = (L2A \ (mu*b + tau*phi'*wh_mat(:,k) -phi'*lh_mat(:,k)));

    %L1 solve
    px_mat(:,k) = phi*x_mat(:,k);
    w_soft = px_mat(:,k) + lh_mat(:,k)/tau;
    w_mat(:,k+1) = wthresh(w_soft,'s',(1/tau));

    %Lagrange update
    l_mat(:,k+1) = lh_mat(:,k) + tau*(px_mat(:,k) - w_mat(:,k+1));
    
    % Combined residual calculation
    c(k+1) = (norm(l_mat(:,k+1)-lh_mat(:,k))^2)/tau + tau*norm(w_mat(:,k+1)-wh_mat(:,k))^2;

    % Restart
    if c(k+1) < eta*c(k)
        a(k+1) = (1+sqrt(1+4*a(k)))/2;
        wh_mat(:,k+1) = w_mat(:,k+1) + (a(k)-1)/(a(k+1))*(w_mat(:,k+1)-w_mat(:,k));
        lh_mat(:,k+1) = l_mat(:,k+1) + (a(k)-1)/(a(k+1))*(l_mat(:,k+1)-l_mat(:,k));
    else
        a(k+1) = 1;
        wh_mat(:,k+1) = w_mat(:,k);
        lh_mat(:,k+1) = l_mat(:,k);
        c(k+1) = c(k)/eta;
    end

    % Residual calculation
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
px = [reshape(px_mat(1:MN,1:k),M,N,k);reshape(px_mat(MN+1:2*MN,1:k),M,N,k)];
w = [reshape(w_mat(1:MN,1:k+1),M,N,k+1);reshape(w_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
wh = [reshape(wh_mat(1:MN,1:k+1),M,N,k+1);reshape(wh_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
l = [reshape(l_mat(1:MN,1:k+1),M,N,k+1);reshape(l_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
lh = [reshape(lh_mat(1:MN,1:k+1),M,N,k+1);reshape(lh_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
a = a(1:k+1);
c = c(1:k+1);
n = n(1:k);
r = r(1:k);
%}

end

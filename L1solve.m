% L1solve: applies TV denoising to solve Ax = b
% Inputs:
%	b = measured image [ M x N ]
%	xin = initial guess [ M x N ]
%	mu = quadratic penalty
%	tau = L1 weight
%	max_iter = maximum number of iterations
% Outputs:
%	x_out = image iterations [ M x N x iter ]
%	w_out = w iterations [ 2M x N x iter ]
%	l_out = Lagrange multiplier iterations [ 2M x N x iter] 
%	px_out = phi*x iterations [ 2M x N x iter] 
%	n1 = ||phi*x - w|| iterations [ iter x 1 ]
%	n2 = ||A*x - b|| / ||b|| iterations [ iter x 1 ]

function [x_out,w_out,l_out,px_out,n1,n2] = L1solve(b,xin,mu,tau,max_iter)

%% Set up variables

% Image size
[M,N] = size(b);
MN = M*N;

% Image space
b = b(:);
x = xin(:);

% Gradient space
w = zeros(2*MN,1);
w_i = zeros(MN,2);
w_0 = zeros(MN,1);
w_soft = zeros(MN,1);
l = zeros(2*MN,1);

% Physics model
A = eye(MN,MN);
AA = A;

%% Create gradient operator
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
phixx = phix'*phix;
phiyy = phiy'*phiy;
phi = [phix;phiy];

%% Augmented Lagrangian algorithm

% Loop variables
iter = 1;
e = 10^-4;          % error tolerance
L2A = AA + (phixx + phiyy)/mu;	% constant matrix in L2 subproblem calculation

% Variables for iterations
x_mat = zeros(MN,max_iter+1);
x_mat(:,1) = x;
w_mat = zeros(2*MN,max_iter+1);
w_mat(:,1) = w;
l_mat = zeros(2*MN,max_iter+1);
l_mat(:,1) = l;
px_mat = zeros(2*MN,max_iter+1);
px_mat(:,1) = phi*x_mat(:,1);

% Stopping critera
n1 = zeros(max_iter+1,1); n2 = n1;
n1(1) = norm(phi*x_mat(:,1)-w_mat(:,1));
n2(1) = norm(A*x_mat(:,1)-b)/norm(b);

%while( ( norm(phi*x-w) > e ) && ( norm(A*x-b)^2 > e ) && (iter <= max_iter) )
while( ( n2(iter) > e ) && ( iter <= max_iter ) )
    
    % Solve L2 subproblem analytically
    d = w_mat(:,iter) + mu*l_mat(:,iter);
    x_mat(:,iter+1) = L2A \ ( (b'*A) + ((d(1:MN)'*phix) + (d(MN+1:2*MN)'*phiy))/mu )';
 
    % Solve L1 subproblem with soft thresholding operator
    for ii = 1:2
        
        w_soft = phi((M*N*(ii-1)+1):(M*N*ii),:)*x_mat(:,iter) - mu*l_mat((M*N*(ii-1)+1):(M*N*ii),iter);
        %w_i(:,ii) = wthresh(w_soft,'s',tau*mu);
	w_i(:,ii) = sign(w_soft).*max(w_0,abs(w_soft) - tau*mu);
        %w_i(:,ii) = w_soft./abs(w_soft).*max(w_0,abs(w_soft) - tau*mu);
        %w_i(:,ii) = wthresh(w_soft,'s',tau*mu);
        
    end
    w_mat(:,iter+1) = [w_i(:,1); w_i(:,2)];

  
    % Update Lagrange multiplier
    l_mat(:,iter+1) = l_mat(:,iter) - (phi*x_mat(:,iter+1) - w_mat(:,iter+1))/mu;
    
    % Update norms and iterations
    iter = iter+1;
    n1(iter) = norm(phi*x_mat(:,iter) - w_mat(:,iter));
    n2(iter) = norm(A*x_mat(:,iter) - b)/norm(b);
    px_mat(:,iter) = phi*x_mat(:,iter);
 
end

% Clear variables and prepare outputs

clear x w l x_temp w_temp w_i w_0 w_soft ii phix phiy;

x_mat = x_mat(:,1:iter);
w_mat = w_mat(:,1:iter);
l_mat = l_mat(:,1:iter);
n1 = n1(1:iter);
n2 = n2(1:iter);
px_mat = px_mat(:,1:iter);

x_out = reshape(x_mat,M,N,iter);
w_out = [reshape(w_mat(1:MN,:),M,N,iter);reshape(w_mat(MN+1:2*MN,:),M,N,iter)];
l_out = [reshape(l_mat(1:MN,:),M,N,iter);reshape(l_mat(MN+1:2*MN,:),M,N,iter)];
px_out = [reshape(px_mat(1:MN,:),M,N,iter);reshape(px_mat(MN+1:2*MN,:),M,N,iter)];

end

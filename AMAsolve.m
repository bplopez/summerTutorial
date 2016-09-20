% AMAsolve: applies the Fast Alternating Minimization Algorithm from the  
% Goldstein paper where the stopping criterion is when subsequent iterations  
% differ by less than 0.5%. 
%
% Call:       u = AMAsolve(b,mu,tau,phi,max_iter)
%
%             Or [u,pres,dres,r,n,a,px,l,lh,w] = AMAsolve(b,mu,tau,phi,max_iter)
%
% Inputs:     b = measured image [ M x N ]
%             mu = strongly complex constant
%             tau = step size (< mu/8)
%             phi = forward finite difference (2D directions) [ 2MN x MN ]
%             max_iter = maximum number of iterations
%         
% Outputs:    u = image iterations [ M x N x iter ]
%             pres = norm of primal residual [ iter x 1 ]
%             dres = norm of dual residual [ iter x 1 ]
%             pu = phi*(image iterations) [ 2M x N x iter]
%             l = Lagrange multiplier [ 2M x N x (iter+1) ]
%             lh = Lagrange multiplier [ 2M x N x (iter+1) ]
%             w = gradient image iterations [ 2M x N x iter ]
%             a = acceleration factor [ (iter+1) x 1]
%             n = norm of image iteration differences [ iter x 1 ]
%             r = norm of residual with input image [ iter x 1 ]

function  [u,pres,dres,r,n,a,pu,l,lh,v] = AMAsolve(b,mu,tau,phi,max_iter)

%% Set up variables

% Image space
[M,N] = size(b);
MN = M*N;
b = b(:);

%% AMA algorithm

% Stopping criteria
e = 0.005;

% Iteration variables initialization
u_mat = zeros(MN,max_iter);
pu_mat = zeros(2*MN,max_iter);
v_mat = zeros(2*MN,max_iter);
l_mat = zeros(2*MN,max_iter+1);
lh_mat = zeros(2*MN,max_iter+1);
a = zeros(max_iter+1,1); a(1) = 1;
n = zeros(max_iter,1);
r = zeros(max_iter,1);
pres = zeros(max_iter,1);       % primal residual ||b-Ax-Bw||^2
dres = zeros(max_iter,1);       % dual residual ||tAB(w-w)||^2

for k = 1:max_iter
        
    % L2 solve
    u_mat(:,k) = b - (phi'*lh_mat(:,k))/mu;
    
    % L1 solve
    pu_mat(:,k) = phi*u_mat(:,k);
    w_soft = pu_mat(:,k) + lh_mat(:,k)/tau;
    v_mat(:,k) = wthresh(w_soft,'s',(1/tau));
    
    % Lagrange update with acceleration
    l_mat(:,k+1) = lh_mat(:,k) + tau*(pu_mat(:,k)-v_mat(:,k));
    a(k+1) = (1+sqrt(1+4*a(k)^2))/2;
    lh_mat(:,k+1) = l_mat(:,k+1) + (a(k) -1)/(a(k+1))*(l_mat(:,k+1) - l_mat(:,k));
   
    % Residuals
    pres(k) = norm(pu_mat(:,k)-v_mat(:,k))^2;
    dres(k) = norm(tau*phi'*(l_mat(:,k+1)-lh_mat(:,k)))^2;
    r(k) = norm(u_mat(:,k)-b)^2;
    
    % Stopping criterion
    if k == 1
        n(k) = 1; 
        continue
    else
        n(k) = norm(u_mat(:,k)-u_mat(:,k-1))/norm(u_mat(:,k));
        if n(k) < e
            break
        end
    end
    
end

%% Clean up variables
u = reshape(u_mat(:,1:k),M,N,k);
pu = [reshape(pu_mat(1:MN,1:k),M,N,k); ...
          reshape(pu_mat(MN+1:2*MN,1:k),M,N,k)];
v = [reshape(v_mat(1:MN,1:k),M,N,k); ...
         reshape(v_mat(MN+1:2*MN,1:k),M,N,k)];
l = [reshape(l_mat(1:MN,1:k+1),M,N,k+1); ...
         reshape(l_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
lh = [reshape(lh_mat(1:MN,1:k+1),M,N,k+1); ...
          reshape(lh_mat(MN+1:2*MN,1:k+1),M,N,k+1)]; 
a = a(1:k+1);
n = n(1:k);
r = r(1:k);

end

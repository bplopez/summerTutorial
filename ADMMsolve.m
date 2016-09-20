% ADMMsolve: applies Fast ADMM algorithm from Goldstein paper where the 
% stopping criterion is when subsequent iterations differ by less than 0.5%. 
%
% Call:       u = ADMMsolve(b,mu,tau,phi,max_iter)
%
%             Or [u,pres,dres,r,n,c,a,pu,l,lh,v,vh] = ADMMsolve(b,mu,tau,phi,max_iter)
%
% Inputs:     b = measured image [ M x N ]
%             mu = strongly complex constant
%             tau = step size
%             phi = forward finite difference (2D directions) [ 2MN x MN ]
%             max_iter = maximum number of iterations
%         
% Outputs:    u = image iterations [ M x N x iter ]
%             pres = norm of primal residual [ iter x 1 ]
%             dres = norm of dual residual [ iter x 1 ]
%             pu = phi*(image iterations) [ 2M x N x iter]
%             l = Lagrange multiplier [ 2M x N x (iter+1) ]
%             lh = Lagrange multiplier [ 2M x N x (iter+1) ]
%             v = gradient image iterations [ 2M x N x (iter+1) ] 
%             vh = gradient image iterations [ 2M x N x (iter+1) ]
%             a = acceleration factor [ (iter+1) x 1]
%             c = combined residual [ (iter+1) x 1 ]
%             n = norm of image iteration differences [ iter x 1 ]
%             r = norm of residual with input image [ iter x 1 ]

function [u,pres,dres,r,n,c,a,pu,l,lh,v,vh] = ADMMsolve(b,mu,tau,phi,max_iter)

%% Set up variables

% Image size
[M,N] = size(b);
MN = M*N;
b = b(:);

% Gradient operator
L2A = mu*eye(MN,MN) + tau*(phi'*phi);	% constant used in L2 subproblem

%% ADMM algorithm

% Stopping criteria
e = 0.005;

% Residual difference factor
eta = 0.999;

% Iteration variables
u_mat = zeros(MN,max_iter+1);
pu_mat = zeros(2*MN,max_iter+1);
v_mat = zeros(2*MN,max_iter+1);
vh_mat = zeros(2*MN,max_iter+1);
v_mat(:,1) = vh_mat(:,1);
l_mat = zeros(2*MN,max_iter+1);
lh_mat = zeros(2*MN,max_iter+1);
l_mat(:,1) = lh_mat(:,1);
a = zeros(max_iter+1,1); a(1) = 1;	% acceleration factor
n = zeros(max_iter,1); 			% ||u-u*||/||u*|| for stopping
r = zeros(max_iter,1);			% residual ||u-b||^2
c = zeros(max_iter+1,1); c(1) = 10^6;	% combined residual
pres = zeros(max_iter,1);       % primal residual ||b-Au-Bv||^2
dres = zeros(max_iter,1);       % dual residual ||tAB(v-vh)||^2

% Iteration
for k = 1:max_iter

    % L2 solve
    u_mat(:,k+1) = (L2A \ (mu*b + tau*phi'*vh_mat(:,k) -phi'*lh_mat(:,k)));

    %L1 solve
    pu_mat(:,k+1) = phi*u_mat(:,k+1);
    w_soft = pu_mat(:,k+1) + lh_mat(:,k)/tau;
    v_mat(:,k+1) = wthresh(w_soft,'s',(1/tau));

    %Lagrange update
    l_mat(:,k+1) = lh_mat(:,k) + tau*(pu_mat(:,k+1) - v_mat(:,k+1));
    
    % Combined residual calculation
    c(k+1) = (norm(l_mat(:,k+1)-lh_mat(:,k))^2)/tau + tau*norm(v_mat(:,k+1)-vh_mat(:,k))^2;

    % Restart
    if c(k+1) < eta*c(k)
        a(k+1) = (1+sqrt(1+4*a(k)))/2;
        vh_mat(:,k+1) = v_mat(:,k+1) + (a(k)-1)/(a(k+1))*(v_mat(:,k+1)-v_mat(:,k));
        lh_mat(:,k+1) = l_mat(:,k+1) + (a(k)-1)/(a(k+1))*(l_mat(:,k+1)-l_mat(:,k));
    else
        a(k+1) = 1;
        vh_mat(:,k+1) = v_mat(:,k);
        lh_mat(:,k+1) = l_mat(:,k);
        c(k+1) = c(k)/eta;
    end

    % Residual calculations
    pres(k) = norm(pu_mat(:,k+1)-v_mat(:,k+1))^2;
    dres(k) = norm(tau*phi'*(v_mat(:,k+1)-vh_mat(:,k)))^2;
    r(k) = norm(u_mat(:,k+1)-b)^2;

    % Stopping criterion
    if k == 1
        n(k) = 1;
        continue
    else
        n(k) = norm(u_mat(:,k+1)-u_mat(:,k))/norm(u_mat(:,k+1));
        if n(k) < e
            break
        end
    end

end

%% Clean up variables
u = reshape(u_mat(:,1:k+1),M,N,k+1);
pu = [reshape(pu_mat(1:MN,1:k+1),M,N,k+1);reshape(pu_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
v = [reshape(v_mat(1:MN,1:k+1),M,N,k+1);reshape(v_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
vh = [reshape(vh_mat(1:MN,1:k+1),M,N,k+1);reshape(vh_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
l = [reshape(l_mat(1:MN,1:k+1),M,N,k+1);reshape(l_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
lh = [reshape(lh_mat(1:MN,1:k+1),M,N,k+1);reshape(lh_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
a = a(1:k+1);
c = c(1:k+1);
n = n(1:k);
r = r(1:k);
pres = pres(1:k);
dres = dres(1:k);

end

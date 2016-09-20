% ADMMsolve_CV: applies the Fast ADMM algorithm from Goldstein paper where 
% the stopping criteria is when the norm of the difference between the current 
% iteration and a provided cross validation (CV) image exceeds the difference 
% between the previous iteration and the CV image. This implementation prevents
% overfitting the noise when denoising the meaured image
%
% Call:       u = ADMMsolve_CV(b,cv,mu,tau,phi,max_iter)
%
%             Or [u,pres,dres,cres,bres,cvres,accel] = ADMMsolve_CV(b,cv,mu,tau,phi,max_iter)
%
% Inputs:     b = measured image [ M x N ]
%             cv = cross validation image [ M x N ]
%             mu = strongly complex constant
%             tau = step size
%             phi = forward finite difference (2D directions) [ 2MN x MN ]
%             max_iter = maximum number of iterations
%         
% Outputs:    u = image iterations [ M x N x iter ]
%             pres = norm of primal residual [ iter x 1 ]
%             dres = norm of dual residual [ iter x 1 ]
%             bres = norm of difference between current iteration and the
%                    provided image to denoise
%             cres = norm of combined residual [ (iter+1) x 1 ]
%             cvres = norm of difference between CV image and current
%                     iteration [ (iter+1) x 1 ]
%             accel = acceleration factor [ (iter+1) x 1]

function [u,pres,dres,cres,bres,cvres,accel] = ADMMsolve_CV(b,cv,mu,tau,phi,max_iter)

%
%% Set up variables

% Image size
[M,N] = size(b);
MN = M*N;
b = b(:);
cv = cv(:);

% Gradient operator
L2A = mu*eye(MN,MN) + tau*(phi'*phi);	% constant used in L2 subproblem

%% ADMM algorithm

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
accel = zeros(max_iter+1,1); accel(1) = 1;	% acceleration factor
cvres = zeros(max_iter,1); 			% ||u-cv|| for stopping
cvres(1) = norm(cv);
bres = zeros(max_iter,1);			% residual ||u-b||^2
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
        accel(k+1) = (1+sqrt(1+4*accel(k)))/2;
        vh_mat(:,k+1) = v_mat(:,k+1) + (accel(k)-1)/(accel(k+1))*(v_mat(:,k+1)-v_mat(:,k));
        lh_mat(:,k+1) = l_mat(:,k+1) + (accel(k)-1)/(accel(k+1))*(l_mat(:,k+1)-l_mat(:,k));
    else
        accel(k+1) = 1;
        vh_mat(:,k+1) = v_mat(:,k);
        lh_mat(:,k+1) = l_mat(:,k);
        c(k+1) = c(k)/eta;
    end

    % Residual calculations
    pres(k) = norm(pu_mat(:,k+1)-v_mat(:,k+1))^2;
    dres(k) = norm(tau*phi'*(v_mat(:,k+1)-vh_mat(:,k)))^2;
    bres(k) = norm(u_mat(:,k+1)-b)^2;

    % Stopping criterion
    cvres(k+1) = norm(cv-u_mat(:,k+1));
    
    if cvres(k+1) >= cvres(k)
        break
    end
    
    %{
    if k == 20
        break
    end
    %}
end

%% Clean up variables
u = reshape(u_mat(:,1:k+1),M,N,k+1);
accel = accel(1:k+1);
cres = c(1:k+1);
cvres = cvres(1:k+1);
bres = bres(1:k);
pres = pres(1:k);
dres = dres(1:k);

end
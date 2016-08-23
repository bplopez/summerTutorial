% AADMMsolve: applies algorithm from Xu paper
%
% Call:       [u,pres,dres,tau,r,n,c,a,pu,l,lh,v,vh] = AADMMsolve(b,mu,tau,phi,max_iter)
%
% Inputs:     b = measured image [ M x N ]
%             mu = strongly complex constant
%             tau = step size (< mu/8)
%             phi = forward finite difference (2D directions) [ 2MN x MN ]
%             max_iter = maximum number of iterations
%         
% Outputs:    u = image iterations [ M x N x iter ]
%             pu = phi*(image iterations) [ 2M x N x iter]
%             l = Lagrange multiplier [ 2M x N x (iter+1) ]
%             lh = Lagrange multiplier [ 2M x N x (iter+1) ]
%             v = gradient image iterations [ 2M x N x (iter+1) ] 
%             vh = gradient image iterations [ 2M x N x (iter+1) ]
%             a = acceleration factor [ (iter+1) x 1]
%             c = combined residual [ (iter+1) x 1 ]
%             n = norm of image iteration differences [ iter x 1 ]
%             r = norm of residual with input image [ iter x 1 ]
%             pres = norm of primal residual [ iter x 1 ]
%             dres = norm of dual residual [ iter x 1 ]
%             tau = stepsize used in each iteration [ iter x 1 ]

function [u,pres,dres,tau,as,bs,r,n,c,a,pu,l,lh,v,vh] = AADMMsolve(b,mu,tau,phi,max_iter)

%
%% Set up variables

% Image size
[M,N] = size(b);
MN = M*N;
b = b(:);

% Gradient operator
L2A = mu*eye(MN,MN) + tau*(phi'*phi);	% constant used in L2 subproblem

%% AADMM algorithm

% Step size restriction
%{
if tau^3 > mu/16
tau = (mu/20)^(1/3);
end
%}

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

% Spectral penalty parameter selection
eC = 0.2;
Tf = 2;
%eT = 0.001;
tau = [tau; zeros(max_iter-1,1)];
%tau = ones(max_iter,1)*tau;
ls_mat = zeros(2*MN,max_iter+1);
as = zeros(max_iter,1);
bs = zeros(max_iter,1);

% Iteration
for k = 1:max_iter

    % L2 solve
    u_mat(:,k+1) = (L2A \ (mu*b + tau(k)*phi'*vh_mat(:,k) -phi'*lh_mat(:,k)));

    %L1 solve
    pu_mat(:,k+1) = phi*u_mat(:,k+1);
    w_soft = pu_mat(:,k+1) + lh_mat(:,k)/tau(k);
    v_mat(:,k+1) = wthresh(w_soft,'s',(1/tau(k)));

    %Lagrange update
    l_mat(:,k+1) = lh_mat(:,k) + tau(k)*(pu_mat(:,k+1) - v_mat(:,k+1));
    
    % Combined residual calculation
    c(k+1) = (norm(l_mat(:,k+1)-lh_mat(:,k))^2)/tau(k) + tau(k)*norm(v_mat(:,k+1)-vh_mat(:,k))^2;

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
    dres(k) = norm(tau(k)*phi'*(v_mat(:,k+1)-vh_mat(:,k)))^2;
    r(k) = norm(u_mat(:,k+1)-b)^2;
 
    % Spectral penalty parameter
    
    % use current iteration of u_mat but previous iteration of vh_mat
    ls_mat(:,k+1) = lh_mat(:,k) + tau(k)*(pu_mat(:,k+1) - vh_mat(:,k));
    
    if mod(k,Tf) == 0 && k > Tf
        
        % Compute spectral stepsizes
        dls = ls_mat(:,k)-ls_mat(:,k-Tf);
        dH = phi*(u_mat(:,k)-u_mat(:,k-Tf));
        dl = lh_mat(:,k)-lh_mat(:,k-Tf);
        dG = vh_mat(:,k)-vh_mat(:,k-Tf);
        
        alphSD = norm(dls)^2/(dls'*dH);
        alphMG = (dls'*dH)/norm(dH)^2;
        betaSD = norm(dl)^2/(dl'*dG);
        betaMG = (dl'*dG)/norm(dG)^2;
        
        if 2*alphMG > alphSD
            as(k) = alphMG;
        else
            as(k) = alphSD - alphMG/2;
        end
        
        if 2*betaMG > betaSD
            bs(k) = betaMG;
        else
            bs(k) = betaSD - betaMG/2;
        end
        
        as(k) = alphSD;
        bs(k) = betaSD;
        % Estimate correlations
        alphC = (dls'*dH)/(norm(dH)*norm(dls));
        betaC = (dl'*dG)/(norm(dG)*norm(dl));
        
        % Update stepsize
        if alphC > eC && betaC > eC
            tau(k+1) = sqrt(as(k)*bs(k));
        elseif alphC > eC && betaC <= eC
            tau(k+1) = as(k);
        elseif alphC <= eC && betaC > eC
            tau(k+1) = bs(k);
        else
            tau(k+1) = tau(k);
        end
        
    else
        tau(k+1) = tau(k);
    end
    
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
tau = tau(1:k);
ls = [reshape(ls_mat(1:MN,1:k+1),M,N,k+1);reshape(ls_mat(MN+1:2*MN,1:k+1),M,N,k+1)];
as = as(1:k);
bs = bs(1:k);

end
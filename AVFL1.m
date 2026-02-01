function [label,loss,all_time]=AVFL1(X,p,lambda,r)
%   X       - Dataset matrix
%   p       - Number of clusters
%   lambda  - Regularization coefficient
%   r       - Step size
%   label   - Clustering result labels
%   loss    - Objective function value
%   all_time- Total algorithm running time (s)

tolerance = 1e-4;
n = size(X, 1);

tic
[W]=initialize_kmeans(n, p);
qz_loga1=initialize_qz_loga(n);
W=W';

for iteration=1:300
        mf = W.^((qz_loga1)+1);
        mf(all(mf == 0, 2), 1) = eps;
        theta = mf*X./(sum(mf,2)*ones(1,size(X,2)));

        dist = pdist2(theta, X);
        W =update_W(qz_loga1,p,n,dist);

        m=qz_loga1+1;
        W3=(W.^ m).*log(W+eps);
        grad=sum((dist.^2) .* W3)+lambda*sign(qz_loga1);
        qz_loga1=max(qz_loga1-r*grad,0);

        objective=compute_objective(qz_loga1,W,dist,lambda);
   
    if iteration > 1
           diff = abs(objective - objective_prev);
           fprintf('Iteration %d: Current objective value = %.6f, Previous objective value = %.6f, Difference = %.6f\n', ...
           iteration, objective, objective_prev, diff);
           if diff < tolerance
           break;
           end
     end
     objective_prev = objective;  % 更新前一次值
    
end

    loss=objective_prev;
    W=W';
    [~,label]=max(W,[],2);
    all_time=toc;
    disp('end')

end


function [W] = initialize_kmeans(n, p)
         W = rand(n, p);
         row_sums = sum(W, 2);
         W = W./row_sums;
end

function [qz_loga] = initialize_qz_loga(n)
         qz_loga = ones(1, n); 
end

function [objective]=compute_objective(qz_loga1,W,dist,lambda)
         m=qz_loga1+1;
         W4=W.^ m;
         objective=sum(sum((dist.^2) .* W4))+lambda*sum(abs(qz_loga1));
end

function [W]=update_W(theta,p,n,dist)
         i1 = find(theta == 0);
         if isempty(i1)
         Q1 = zeros(p, n);  
         else
         [~, j] = min(dist(:, i1));
         Q1 = zeros(p,n);         
         linear_idx = (i1 - 1) * p + j;
         Q1(linear_idx) = 1;        
         end

         i2 = find(theta ~= 0);
         if isempty(i2)
         Q3 = zeros(p, n);  
         else
         dist_i2 = dist(:, i2);
         theta_i2=theta(:, i2);
         m=max(-2./(theta_i2),-80);
         tmptmp=((dist_i2+eps).^m);
         tmp3_2=(tmptmp)./(eps+ones(p,1)*(sum(tmptmp)));
         Q3 = zeros(p,n);
         Q3(:, i2) = tmp3_2;
         end

         W=Q3+Q1;
end
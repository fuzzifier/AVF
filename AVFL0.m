function [label,loss,all_time]=AVFL0(X,p,lambda,r)
%   X       - Dataset matrix
%   p       - Number of clusters
%   lambda  - Regularization coefficient
%   r       - Step size
%   label   - Clustering result labels
%   loss    - Objective function value
%   all_time- Total algorithm running time (s)

n = size(X, 1);
droprate_init = 0.5; 
temperature= 2/3;   
limit_a=-0.1;
limit_b=1.1;
epsilon=eps;
prev_loss = inf;
prev_loss1=inf;
tolerance = 1e-4;
beta1 = 0.9;
beta2 = 0.999;
epsilon_adam = 1e-8;

tic
[W]=initialize(n, p);
qz_loga1=initialize_qz_loga(droprate_init, n);
W=W';
u=generate_uniform_matrix(p, n, epsilon);
d = mask(u,p, qz_loga1, temperature, limit_a, limit_b);

    for iteration=1:700
        mf = W.^((2*d)+1+epsilon);
        mf(all(mf == 0, 2), 1) = eps;
        theta = mf*X./(sum(mf,2)*ones(1,size(X,2)));

        dist = pdist2(theta, X)+eps;
        W = update_membership_matrix(dist, d, p, n,epsilon);

        m = zeros(size(qz_loga1));
        v = zeros(size(qz_loga1)); 
        t = 0; 
        qz_loga1 = dlarray(qz_loga1, 'CB');
           for iter = 1:40
               [loss, grad] = dlfeval(@(qz_loga1) objective_function(W,dist, u,p,  qz_loga1, lambda, limit_a, limit_b, temperature, epsilon), qz_loga1);
               t = t + 1;
               m = beta1 * m + (1 - beta1) * grad;
               v = beta2 * v + (1 - beta2) * grad.^2;
               m_hat = m / (1 - beta1^t);
               v_hat = v / (1 - beta2^t);
               qz_loga1 = qz_loga1 - r * m_hat ./ (sqrt(v_hat) + epsilon_adam);
               if ~isinf(prev_loss1)
                    if abs(extractdata(loss) - prev_loss1) < 1e-6
                         break;
                    end
               end
                    prev_loss1 = extractdata(loss);
              
               if iter<40
               u=generate_uniform_matrix(p, n, epsilon);
               end
           end
        qz_loga1 = extractdata(qz_loga1);
        d = mask(u,p, qz_loga1, temperature, limit_a, limit_b);
        mf = W.^((2*d)+1+epsilon);
        mf(all(mf == 0, 2), 1) = eps;

        loss2=objective_function2(dist, mf, qz_loga1, lambda, limit_a, limit_b, temperature, epsilon);
        fprintf('Iteration %d: Current objective function value = %.4f\n', iteration, loss2);

        if ~isinf(prev_loss)
            if abs(loss2 - prev_loss) < tolerance
                fprintf('Converged at iteration %d with loss difference: %.4e\n', iteration, abs(loss2 - prev_loss));
                break;
            end
        end
        prev_loss = loss2;
    end 

   loss=loss2;
   W=W';
   [~,label]=max(W,[],2);
   all_time=toc;
   disp('end')
end


function [W] = initialize(n, p)
          W = rand(n, p);
          row_sums = sum(W, 2);
          W = W./row_sums;
end

function [qz_loga1] = initialize_qz_loga(droprate_init, n)
         mean_value = log(1 - droprate_init) - log(droprate_init);
         std_dev = 1e-2;
         qz_loga1 = mean_value + std_dev * randn(1, n);
         min_val = log(1e-2); 
         max_val = log(1e2);   
         qz_loga1 = min(max(qz_loga1, min_val), max_val);  
end

function [regularization] = regularization(qz_loga1, lambda, limit_a, limit_b, temperature, epsilon) 
         xn = (0 - limit_a) / (limit_b - limit_a);
         logits1 = log(xn) - log(1 - xn);
         sigmoid_value = 1 ./ (1 + exp(-(logits1 * temperature - qz_loga1)));
         cdf_qz = min(max(sigmoid_value, epsilon), 1 - epsilon);
         regularization = lambda*sum((1-cdf_qz), 'all');
end

function [u] = generate_uniform_matrix(p, n, epsilon)
         u1= epsilon + (1 - 2 * epsilon) * rand(1, n);
         u= repmat(u1, p, 1);
end

function [mask] = mask(x,p, qz_loga1, temperature, limit_a, limit_b)
         qz_loga= repmat(qz_loga1, p, 1);
         logits = log(x) - log(1 - x) + qz_loga;
         y = 1 ./ (1 + exp(-logits ./ temperature));
         z = y * (limit_b - limit_a) + limit_a;
         mask = min(max(z, 0), 1);
end

function [loss, grad] = objective_function(W,dist, u,p, qz_loga1, lambda, limit_a, limit_b, temperature, epsilon)
         d = mask(u,p, qz_loga1, temperature, limit_a, limit_b);
         mf = W.^((2*d)+1+epsilon);
         reg = regularization(qz_loga1, lambda, limit_a, limit_b, temperature, epsilon);
         loss = sum(sum((dist.^2) .* mf)) + reg;
         grad = dlgradient(loss, qz_loga1);
end

function [loss1] = objective_function2(dist, mf, qz_loga1, lambda, limit_a, limit_b, temperature, epsilon)
         reg = regularization(qz_loga1, lambda, limit_a, limit_b, temperature, epsilon);
         loss1 = sum(sum((dist.^2) .* mf)) + reg;
end

function [W] = update_membership_matrix(dist, d, p, n,epsilon)
         tmp1 = repmat((dist(:)).', p, 1);
         tmp2 = kron(dist, ones(1, p));
         tmp3 = 2./(2 * d + epsilon);
         tmp3_1 = kron(tmp3(:).', ones(p, 1));
         W = reshape(((sum((tmp1 ./ tmp2).^tmp3_1, 1)).^(-1)).', p, n);
end
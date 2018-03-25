function X = project_to_simplex(Y)
% taken from https://arxiv.org/pdf/1309.1541.pdf
% projects Y to a D = size(Y,1) dimensional simplex
% after projection, we will have that X_i \in [0,1] for all i
% and sum(X) = 1

[N,D] = size(Y);
X = sort(Y,2,'descend');
Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);



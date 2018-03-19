function [  ] = print_matrix( matrix, xlabel, ylabel, title )
if nargin > 3
    fprintf('%s\n', title);
end%if

[m, n] = size(matrix);
fprintf('\t');
for i = 1:n
    fprintf('%s=%d\t', xlabel, i);
end%for
fprintf('\n');
for i = 1:m
    for j = 1:n
        if j == 1
            fprintf('%s(%d)\t',ylabel, i);
        end%if
        fprintf('%.4f\t', matrix(i,j))
    end%for
    fprintf('\n');
end%for
fprintf('\n');
end%func
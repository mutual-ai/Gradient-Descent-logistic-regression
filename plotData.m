function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

a = find (y == 1);
b = find (y == 0);

plot(X(a,1),X(a,2),'k+','MarkerSize',10);
plot(X(b,1),X(b,2),'ko','MarkerSize',10);

hold off;

end

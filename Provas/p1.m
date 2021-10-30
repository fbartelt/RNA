%% Questao 1
X1 = [-1 -1 1];
X2 = [0 0 1];
X3 = [2 2 1];
X = [X1;X2;X3]
%y = [[0 0];[0 1];[1 1]]
%w = pinv(X)*y
b1 = -1
a1 = -1
b2 = 1.7
a2 = -1.7/1.5

scatter([-1 0 2],[-1 0 2])
x = linspace(-2,3)
w11 = 1.2;
w21 = 1;
w01 = -1.1;
w12 = 1.2;
w22 = 1;
w02 = 1.6;
y1 = -w11/w21*x+w01/21;
y2 = -w12/w22*x+w02/w22;
hold on
plot(x,y1)
plot(x,y2)
hold off

%% Questao 3
% w = [2;-0.4;0.2]
% x = [0.9; 1.5; 1]
% eta = 0.1
% y = -1
% y_hat = x'*w
% y_hat = y_hat/abs(y_hat)
% err = y - y_hat
% w = w + eta*err*x

%% Questao 4
% w1 = [-1;-1;4]
% w2 = [-0.5;-1;3]
% w3 = [-1.5;-1;11]
% w4 = [1;-1;0]
% A = [-4.7;2;1]
% B = [2.1;-4.3;1]
% C = [3.9;4.2;1]
% 
% w = [w1 w2 w3 w4]
% for i = 1:4
%     fprintf('\n\nw%d\n', i)
%     fprintf('A:%d\n',sign(A'*w(:, i)))
%     fprintf('B:%d\n',sign(B'*w(:, i)))
%     fprintf('C:%d\n',sign(C'*w(:, i)))
%     disp((sign(A'*w(:, i))==sign(B'*w(:, i)))&&sign(C'*w(:, i))~=sign(A'*w(:, i)))
% end
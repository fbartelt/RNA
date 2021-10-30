%% questao 1
%V F F ? F

% z = [0.1 -0.6 0.5; -0.6 0 -1; -0.9 -0.5 -0.5]
% w = [1.1; 0.9; -1.6; -1.8]
% x = [-3 -3.5]
% 
% xaug = [x 1]
% H = tanh(xaug * z)
% Haug = [H 1]
% Y = Haug*w

%% questao 2

% u1 = [2;2]
% u2 = [4;4]
% x = [3.4; 6.3]
% 
% h1 = norm(x-u1)^2

%% questao 3

u = [pi/2 3*pi/2] + 0.4

x = linspace(0,2*pi,1e3)'
y = 9.4*sin(x-0.4)
H = exp(-(x-u).^2)
Haug = [H ones(1,length(x))']
w = pinv(Haug)*y

x1 = 2.1
h1 = exp(-(x1-u).^2)
h1aug = [h1 1]
y = h1aug*w

%% questao 4

% x1 = [1 1; 0 1; 0 0 ; 1 0]
% y1 = [0;1;0;1]
% u1 = [1 1]
% u2 = [0 0]
% 
% b = 2.8404;
% w = -2.5018*[1; 1];
% for i=1:4
%     x = x1(i,:)
%     h1 = exp(-norm(x-u1).^2);
%     h2 = exp(-norm(x-u2).^2);
%     H = [h1 h2]
%     y = H*w +b
%     round(y)
% end
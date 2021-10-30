%% Questao 1
%     z10  z20   z11 z21    z12  z22
Z = [[-0.2 0.2];[0.5 0.1];[-0.5 -0.3]]
%    w30  w31   w32
w = [0.3; 0.5; -0.1]
alpha = 0.1;

X = [1 1.2 2.5]
y = 3

u = [1 tanh(X * Z)]
h = u * w

delta_w = (y-h)
w_n = w + alpha * (delta_w * u).'
delta_Z = (w * delta_w) .* sech(u).^2
Z_n = Z + alpha * X * (delta_Z(:, 2:end))
round(Z_n(3,1), 5)
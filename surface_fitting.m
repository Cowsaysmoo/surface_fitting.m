%%%Jared Homer, Alex Stephens, Tracey Gibson
clear;clc;

x = linspace(-8,8,30);
y = linspace(-8,8,30);

[x, y] = meshgrid(x,y);

z_samples = sin(sqrt(x.^2 + y.^2)) ./ sqrt(x.^2 + y.^2);

% Normalize samples
[x_n, ps_x] = mapminmax(x, 0, 1);
[y_n, ps_y] = mapminmax(y, 0, 1);
[z_n, ps_z] = mapminmax(z_samples, 0, 1);

% Number of samples
N = size(z_samples,1) * size(z_samples,2);

% Hidden Units in single layer
H = 100;

% Learning Rate
eta = 0.08;

% Initialize weights to random values between -0.01 and 0.01
w = -0.01 + (0.01 - (-0.01)) * rand(2,H);
v = -0.01 + (0.01 - (-0.01)) * rand(H,1);

% Preallocate memory
hidden_layer = zeros(H,1);
d_w = zeros(2,H);
input = zeros(2,1);

for iter = 1:1000
    err = 0; % initialize err to 0 for summing error
    for i = 1:N
        % Randomly select input point
        selection = round(1 + (N - 1) * rand());
        input = [
            x_n(selection);
            y_n(selection)
            ];
        target = z_n(selection);
        
        % Calculate hidden layer output
        for h = 1:H
            w_h = w(:,h);
            % sigmoid function
            hidden_layer(h) = 1 / (1 + exp(-(w_h' * input)));
        end
        
        % Calculate output from hidden layer
        z_out = v(:,1)' * hidden_layer;
        err = err + abs(target - z_out);
        
        % Calculate change in v
        d_v = eta * (target - z_out) * hidden_layer;
        
        % Calculate change in w
        for h = 1:H
            sum = (target - z_out) * v(h);
            d_w(:,h) = eta * sum * hidden_layer(h) * (1 - hidden_layer(h)) * input;
        end
        
        v = v + d_v;
        w = w + d_w;
    end
end

x_test = linspace(-8,8,30);
y_test = linspace(-8,8,30);

[x_test,y_test] = meshgrid(x_test,y_test);

x_t_normal = mapminmax("apply", x_test, ps_x);
y_t_normal = mapminmax("apply", y_test, ps_y);

input_test = [
    reshape(x_t_normal,[1,size(x_test,1)*size(x_test,2)]);
    reshape(y_t_normal,[1,size(y_test,1)*size(y_test,2)])
];

hidden_layer_test = 1 ./ (1 + exp(-(w' * input_test)));
output_normalized = v' * hidden_layer_test;
output_normalized = reshape(output_normalized, [size(x_test,1), size(x_test,2)]);
output = mapminmax("reverse", output_normalized, ps_z);

figure(1);
clf;
surf(x_test,y_test,output);
shading("interp");

figure(2);
clf;
surf(x,y,z_samples);
shading("interp");
%%%Jared Homer, Alex Stephens, Tracey Gibson
clear;clc;

% Create samples
x = linspace(-8,8,10);
y = linspace(-8,8,10);

[x, y] = meshgrid(x,y);

z_samples = sin(sqrt(x.^2 + y.^2)) ./ sqrt(x.^2 + y.^2);
%z_samples = sqrt(x.^2 + y.^2);

% Normalize samples
[x_n, ps_x] = mapminmax(x, 0, 1);
[y_n, ps_y] = mapminmax(y', 0, 1);
y_n = y_n';
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

count = 0;

for iter = 1:3000
    err = 0; % initialize err to 0 for summing error
    for i = 1:N
        % Randomly select input point
        selection_i = round(1 + (size(z_n,1) - 1) * rand());
        selection_j = round(1 + (size(z_n,2) - 1) * rand());
        input = [
            x_n(selection_i, selection_j);
            y_n(selection_i, selection_j)
            ];
        target = z_n(selection_i, selection_j);
        
        % Calculate hidden layer output
        sum = 0;
        for h = 1:H
            w_h = w(:,h);
            % sigmoid function
            hidden_layer(h) = 1 / (1 + exp(-(w_h' * input)));
        end
        
        % Calculate output from hidden layer
        z_out = v' * hidden_layer;
        err = err + abs(target - z_out);
        
        % Calculate change in v
        d_v = eta * (target - z_out) * hidden_layer;
        
        % Calculate change in w
        for h = 1:H
            sum = (target - z_out) * v(h);
            d_w(:,h) = eta * sum * hidden_layer(h) * (1 - hidden_layer(h)) * input;
        end
        
        % Update weights
        v = v + d_v;
        w = w + d_w;
        
        count = count + 1;
        v_history(count) = v(20, 1);
        w_history(count) = w(1, 16);
    end
    % Track error for plotting
    err_history(iter) = err/(N*1.0);
    disp(iter);
end

% Produce test samples
x_test = linspace(-8,8,10);
y_test = linspace(-8,8,10);

[x_test,y_test] = meshgrid(x_test,y_test);

% Normalize test samples
x_t_normal = mapminmax("apply", x_test, ps_x);
y_t_normal = mapminmax("apply", y_test', ps_y);
y_t_normal = y_t_normal';

input_test = [
    reshape(x_t_normal,[1,size(x_test,1)*size(x_test,2)]);
    reshape(y_t_normal,[1,size(y_test,1)*size(y_test,2)])
];

% Calculate output based on learned weights
hidden_layer_test = 1 ./ (1 + exp(-(w' * input_test)));
output_normalized = v' * hidden_layer_test;
output_normalized = reshape(output_normalized, [size(x_test,1), size(x_test,2)]);
output = mapminmax("reverse", output_normalized, ps_z);

% Surface plots
figure(1);
clf;
subplot(1,2,1);
surf(x_test,y_test,output);
shading("interp");
title("Trained NN Output");
subplot(1,2,2);
surf(x,y,z_samples);
shading("interp");
title("Ideal Output");

% Plot error for analysis of accuracy and convergence
figure(2);
clf;
plot(err_history);
title("Error over time");
disp(min(err_history));

% Plot weight histories of selected weights
figure(3);
clf;
subplot(1,2,1);
plot(w_history);
title("w(1,16)");
subplot(1,2,2);
plot(v_history);
title("v(20,1)");
%%%Jared Homer, Alex Stephens, Tracey Gibson
clear all;
clc;

X = (-8:0.1:8);
Y = (-8:0.1:8);

[X_n,ps] = mapminmax(X, 0, 1);
[Y_n,ls] = mapminmax(Y, 0, 1);
input = [X_n; Y_n];

[X_m,Y_m] = meshgrid(X,Y); %Training Samples
R = (sin(sqrt(X_m.^2+Y_m.^2))./sqrt(X_m.^2+Y_m.^2)); %Training Samples

[R_n,ts] = mapminmax(R, 0, 1);

N=size(input,2);

H=120;  
eta=0.08;
M = 30;  %30

z=ones(H,1);
w=ones(2,H); 
v=ones(H,2);  
delta_w=ones(2,H);
delta_v=ones(H,1);

for k=1:2
    for h=1:H
        w(k,h) = (0.01-(-0.01))*rand()+(-0.01);
    end
end
for k=1:2
    for h=1:H
        v(h,k) = (0.01-(-0.01))*rand()+(-0.01);
    end
end
count=0;
for j=1:M
    err=0;
    for t=1:N
         
        c=(N-1)*rand()+1;
        c=round(c);
        
        g=(N-1)*rand()+1;
        g=round(g);
        
        a=input(1,c);
        b=input(2,g);
        
        x=[a; b];
        r=[R_n(c,g); R_n(c,g)];
        for h=1:H
            w_h=w(:,h);
            z(h)=1/(1+exp(-(w_h'*x)));
        end
        for k=1:2
            y(k)=v(:,k)'*z;
            err=err+abs(r(k)-y(k));
        end
        for k=1:2
            delta_v(:,k)=eta*(r(k)-y(k))*z;
        end
        for h=1:H
            sum=0;
            for k=1:2
                sum=sum+(r(k)-y(k))*v(h,k);
            end
            delta_w(:,h)=eta*sum*z(h)*(1-z(h))*x;
        end
        v=v+delta_v;
        w=w+delta_w;

        count = count + 1;
        v_history(count)=v(5,1);
        w_history(count)=w(1,4);
    end
    
    err_history(j)=err/(N*1.0); % save the history of error.
    disp(j)
end

figure;
plot(v_history);
legend ('v(10,1)');
figure;
plot(w_history);
legend('w(1,15)');
figure;
plot(err_history);
legend('err');


X_t=-8:0.1:8;   % generate the test samples 
Y_t=-8:0.1:8;   % generate the test samples 

X_n = mapminmax('apply',X_t,ps); % normalize the test samples
Y_n = mapminmax('apply',Y_t,ls); % normalize the test samples

[X_t,Y_t] = meshgrid(X_t,Y_t);

X_a = [X_n;Y_n];

Z=1./(1+exp(-(w'*X_a)));

R_n1=v(:,1)'*Z;
R_n2=v(:,2)'*Z;

figure;
R=mapminmax('reverse',R_n2,ts);  % reverse the normalization
R2=(sin(sqrt(X_t.^2+Y_t.^2))./sqrt(X_t.^2+Y_t.^2));     

surf(X_t, Y_t, R);
hold on

%surf(X_t, Y_t, R2);
%plot3(X, Y, R, 'k*')
shading interp

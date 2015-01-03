T = 4.0;             %Length of signal in seconds
dt = 0.001;          %Time step size

rms = 0.5;           % RMS signal level
limit = 5;           % 5 Hz upper limit
RandomSeed=37;		 
% Generate bandlimited white noise
[x,X] = genSignal(T,dt,rms,limit,RandomSeed);

Nt = length(x);  %
t = dt*[0:Nt-1]; %

%% LIF Neuron Model Parameters
tau_ref = 0.002;  %
tau_rc  = 0.020;  %
x0 = 0;           % Firing rate at x=x0 is a0  
a0 = 40;          
x1 = 1.0;         % Firing rate at x=x1 is a1
a1 = 150;

%
eps = tau_rc/tau_ref;   
r1 = 1/(tau_ref*a0);
r2 = 1/(tau_ref*a1);
f1 = (r1-1)/eps;
f2 = (r2-1)/eps;
alpha = (1.0/(exp(f2)-1) - 1.0/(exp(f1)-1))/(x1-x0); 
x_threshold = x0-1/(alpha*(exp(f1)-1));               
Jbias = 1-alpha*x_threshold;    

% Simulate the two neurons (use your own function from part 3)
spikes = two_neurons(x,dt,alpha,Jbias,tau_rc,tau_ref);

freq = [0:Nt-1]/T - Nt/(2.0*T);  %
omega = freq*2*pi;             %

r = spikes(:,1)' - spikes(:,2)';     %
R = fftshift(fft(r));          %

sigma_t = 0.025;               %
W2 = exp(-omega.^2*sigma_t^2);  %
W2 = W2/(sum(W2));             %

CP = X.*conj(R);             % 
WCP = convn(CP,W2,'same');   %
RP = R.*conj(R);             % 
WRP = convn(RP,W2,'same');   %
XP = X.*conj(X);             % 
WXP = convn(XP,W2,'same');   %

H = WCP./WRP;                %

h = real(fftshift(ifft(ifftshift(H))));  %

XHAT = H.*R;                 %

xhat = real(ifft(ifftshift(XHAT)));     %



figure(1);clf;
subplot(1,2,1);
plot(freq,sqrt(XP)); %
legend('???');
xlabel('???');
ylabel('???');

subplot(1,2,2);
plot(freq,sqrt(RP)); % 
legend('???');
xlabel('???');
ylabel('???');


figure(2);clf;
subplot(1,2,1);
plot(freq,real(H));   %
xlabel('???');
title('???','FontSize',16);
xlim([-50,50]);

subplot(1,2,2);
plot(t-T/2, h);
title('???','FontSize',16);
xlabel('???');
xlim([-0.5, 0.5]);


figure(3);clf;
plot(t,r,'LineWidth',2,'Color',[.8 .8 .8]); %
hold on;
plot(t,x,'LineWidth',2);   %
plot(t,xhat);              %
title('???','FontSize',16);
legend('???','???','???');
xlabel('???');
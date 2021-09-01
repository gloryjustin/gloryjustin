%% ECSE 6180 HW 4
% Glory Justin 11/9/2020
%% 1. Problem 14.2
% For the system with Xeq as shown in equation 14.18
% 
% $$X_1 = X_2 = \frac{X}{2}\\X_{eq} = \frac{X^2}{4}\left(\frac{2}{X}+\frac{2}{X}+\frac{1}{X_3}\right)\\= 
% \frac{X^2}{4}\left(\frac{4}{X}+\frac{1}{X_3}\right)$$
% 
% $$\frac{1}{X_3} = -B\\X_{eq} = \frac{X^2}{4}\left(\frac{4}{X}-B\right)\\= 
% X - \frac{BX^2}{4}\\= X\left(1 - \frac{BX}{4}\right)$$
%% 2. Problem 14.3
% The active power flow from bus 1 to bus 2 is given by
% 
% $$P = \frac{V^2}{X/n}\sin(\delta/n)\\= \frac{nV^2}{X}\sin(\delta/n)$$
% 
% To find the limit as n approaches infinity,
% 
% $$\lim_{n\to\infty}P = \lim_{n\to\infty}\left(\frac{nV^2}{X}\sin(\delta/n)\right)\\= 
% \frac{nV^2}{X}(\delta/n)\\= (V^2/X)\delta$$
%% 3. Problem 14.5
% $$V_o = 1.0 pu \text{ and} V_{ref} = 1.03 pu\\V = 1.0+0.02I_c\\V=1.03-0.04I_c\\I_c 
% = \frac{0.03}{0.06} = 0.2 pu\\V= 1.0+0.02\times 0.2 = 1.004 pu$$
% 
% From the above, the SVC Var reserve of 0.2pu is not exceeded. Also since the 
% current is positive, it shows that the SVC is operating in the capacitive region.
% 
% $$V_o=1.04 pu \text{ and } V_{ref}= 1.03 pu\\V= 1.04+0.02I_c\\V=1.03-0.04I_c\\I_c 
% =- \frac{0.01}{0.06} = -0.1667pu\\V= 1.04-0.02\times 0.1667 = 1.0367 pu$$
% 
% From the above, the SVC Var reserve of 0.2pu is not exceeded. Also since the 
% current is negative, it shows that the SVC is operating in the inductive region.
%% 4. Problem 14.6
% Using the s_simu command in PST, it shows that without an SVC, the system 
% is stable for fault clearing time of 0.13, and unstable for 0.14 and 0.15 sec. 
% The following plots were generated for these clearing times without the SVC, 
% 0.13, 0.14 and 0.15 sec respectively.
% 
% 
% 
% 
% 
% With an SVC with control range plus or minus 200 MVar, the system is also 
% stable for 0.13, 0.14 but unstable for 0.15 sec clearing times. The following 
% plots were generated for the 0.13, 0.14 and 0.15 sec clearing times respectively.
% 
% 
% 
% 
% 
% With an SVC of control range plus or minus 800 MVar, the system is stable 
% for all 3 clearing times. The following plots were generated for 0.13, 0.14 
% and 0.15 sec clearing times respectively.
% 
% n
% 
% 
%% 5. Problem 14.7

Vm= 1.0; theta_s = 110*pi/180; theta_r=70*pi/180; X_L= 0.1; jay= sqrt(-1);
Vs= Vm*exp(jay*theta_s); Vr= Vm*exp(jay*theta_r);
I= (Vs-Vr)/(jay*X_L) % current in the transmission line
P= Vm^2*sin(theta_s-theta_r)/X_L % the power transfer
V_L= I*(jay*X_L) 
%% 
% For figure 14.68b, the voltages, currents and power transfer are as follows

X_c= -0.05;
I= (Vs-Vr)/(jay*(X_c+X_L)) % current in the transmission line
P= Vm^2*sin(theta_s-theta_r)/(X_L+X_c) % the power transfer
V_L= I*(jay*X_L)
V_c= I*(jay*X_c)
%% 
% For figure 14.68c, the voltages, currents and power transfer are as follows

Vq=0.1;
I= (Vs-Vq-Vr)/(jay*X_L) % current in the transmission line
P= (Vm^2*sin(theta_s-theta_r)+Vm*Vq*sin(theta_s-theta_r))/(X_L) % the power transfer
V_L= I*(jay*X_L)
%For the plots
t = linspace(0,5,892);
% plot all current and voltage phasors, calculate power transfer
Vs = cos(t+(110*pi/180));
Vr = cos(t+(70*pi/180)); 
figure
plot(t, 6.84*cos(t+90),t,0.684*cos(t+180),t,Vs,t,Vr)
xlabel('time')
ylabel('current/voltage (I/V)')
title('No series compensation')
legend('I','V_L','V_s','V_r')
% series compensation
Isc = Vs.*Vr.*sind(40)/.05;
figure
plot(t, Isc,t,Vs, t,Vr)
xlabel('time')
ylabel('current/voltage (I/V)')
title('Series Compensation')
legend('I','V_s','V_r')
% VSR 

Vq = 0.1;
Ivsc =5.84*cos(t+90);
figure
plot(t, Ivsc, t,Vs, t,Vr)
xlabel('time')
ylabel('current/voltage (I/V)')
title('With VSC Compensation');legend('current','V_s','V_r')
%% 6. Problem 14.8

Vs= Vm*exp(jay*theta_s); Vr= Vm*exp(jay*theta_r); I = (Vs-Vr)/(X_L+X_c);
Vc= I*X_c;
V_A= Vs+jay*Vc              % Voltage at point A
V_A_mag= abs(V_A)            % voltage magnitude at point A
V_L = I*jay*(X_c+X_L/2)                 % voltage at mid-point of X_L
V_L_mag= abs(V_L)               % voltage magnitude at mid-point of A
%% 
% The voltage profile for the system is as follows
% 
% 

V_B = Vs-I*jay*X_L/2             % Voltage at point B
V_B_mag = abs(V_B)              % Voltage magnitude at point B
V_C = Vr+I*jay*(X_L/2)      % Voltage at point C
V_C_mag = abs(V_C)              % Voltage magnitude at point C
%% 
% The voltage profile for the configurtion in b is as follows
% 
% 
% 
% To plot the voltage profiles in MATLAB

% Voltage Profiles
ya = [Vm V_A_mag V_L_mag Vm];
xa = [1 2 3 4];
yb = [Vm V_B_mag V_C_mag Vm];
xb = [1 2 3 4];
figure
plot(xa ,ya, '-o')
title('V profile part a, 1)Vs 2)V_A 3)V_L 4)Vr')
figure
plot(xb ,yb, '-o')
title('V profile part b, 1)Vs 2)V_B 3)V_C 4)Vr')
%% 7. Problem 14.10
% (a) For voltage regulation mode

V_pq = 0.1*exp(jay*theta_s);    % Vpq has the same phase angle as the source voltage
Ise = (Vs+V_pq-Vr)/(jay*X_L);
Ppq = real(V_pq*Ise)             % The required active power circulation between the series and shunt VSCs
Qpq = -imag(V_pq*Ise)            % The reactive power injection by the series VSC
%% 
% (b) For the SSSC mode, there's no power circulation.
% 
% (c) For the phase shifting mode, the phase angle of Vpq is chosen such that 
% the magnitude of Vseff = Vs + Vpq remains the same. Thus, let the phase angle 
% of Vpq be 202.85 degrees.

theta_pq= 202.85*pi/180;
Vpq= 0.1*exp(jay*(theta_pq));
Vseff = Vs + Vpq
Vseff_mag = abs(Vseff)
Ise = (Vs+Vpq-Vr)/(jay*X_L);
Ppq = real(Vpq*conj(Ise))             % The required active power circulation between the series and shunt VSCs
Qpq = imag(Vpq*conj(Ise))            % The reactive power injection by the series VSC
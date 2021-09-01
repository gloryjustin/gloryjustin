%% ECSE 6180 Homework 7
%% Glory Justin 12/10/2020
% 1. Problem 10.1

t = [0:1/1440:23/1440];
N = 1440/60;
x = 100*sqrt(2)*cos(2*pi*60*t + pi/4);
X_phasor = fft(x)
X_mag = abs(X_phasor)/(N/2)
X_rms = X_mag/sqrt(2)
X_ang = angle(X_phasor)*180/pi
% 2. Problem 10.4

t_more = [0:1/1440:(23+24)/1440];
X_mag = [];
X_phase = [];
for i = 1:24
  x = 100*sqrt(2)*cos(2*pi*59*(t_more(i:i+23)) + pi/4);
  X_phasor = fft(x);
  X_mag = [X_mag abs(X_phasor(2))];
  X_phase = [X_phase angle(X_phasor(2))*180/pi];
end
X_mag = X_mag/(24/2)/sqrt(2)
X_phase
figure, polar(X_phase*pi/180,X_mag)
% 3. Problem 10.8
% Using the PMU Simulator, the positive sequence Voltage magnitude is determined 
% as 0.028 pu. The following plots were generated.
% 
% For phase A, the magnitude is plotted as follows
% 
% 
% 
% Phase B
% 
% 
% 
% Phase C
% 
% 
% 
% Positive Sequence magnitude and angle
% 
% 
% 
% Negative sequence magnitude and angle
% 
% 
% 
% Zero sequence magnitude and angle
% 
% 
% 4. Problem 10.10

% use PST function s_simu to simulate the 12-machine system in the data file 
% data_emwave_1.m
% Change line reactance to 0.2 pu


% commands for plotting
%set_font_size

% one-second plot
figure, plot(t(1:141),mac_spd(1:11,1:141),[0 1],[0.9999 0.9999])
%% 
% 

xlabel('Time (sec)'), ylabel('Frequency(pu)')

% five-second plot
figure, plot(t,mac_spd(1:11,:))
xlabel('Time (sec)'), ylabel('Frequency(pu)')
%% 
% 


%reset_font_size

figure, mesh(t,[1:1:11],mac_spd(1:11,:)) % use rotate button for a better display 
                                  % of the time response 
xlabel('Time (sec)'), ylabel('machine number'), zlabel('Machine Speed (pu)')
%% 
% 
% 
% With a change in X, k=l/X becomes 0.5 pu-length/pu-reactance. So the speed 
% of the wave propagation becomes

c = sqrt(0.5*377/120)    % in pu-length/s
% 5. Problem 10.13

% line parameters (345 kV), per unit on 100 MVA base
% S to F
r_12 = 0.00003;
x_12 = 0.00042;
b_12 = 0.0;
% F to E
r_32 = 0.00251;
x_32 = 0.0346;
b_32 = 0.592/2;
% raw PMU data
% Bus 1 is S, Bus 2 is F, Bus 3 is E
jay = sqrt(-1);
V1p = 206368.1/345000*sqrt(3)*exp(jay*-(15.912)*pi/180);
V3p = 204659.1/345000*sqrt(3)*exp(jay*-24.12*pi/180);
I12p = 380.4/167.3479*exp(jay*149.856*pi/180);
Is = 1.19; % scaling factor
I32p = (978.4/Is)/167.3479*exp(jay*159.488*pi/180);

% A matrix set up
A = [ ...
    1 0 0 0 0 0 0 0 0 0; ...
    0 1 0 0 0 0 0 0 0 0; ...
    0 0 0 0 1 0 0 0 0 0; ...
    0 0 0 0 0 1 0 0 0 0; ...
    0 0 0 0 0 0 1 0 0 0; ...
    0 0 0 0 0 0 0 1 0 0; ...
    0 0 0 0 0 0 0 0 1 0; ...
    0 0 0 0 0 0 0 0 0 1; ...
    (1-x_12*b_12) -r_12*b_12 -1 0 0 0 -r_12 x_12 0 0; ...
     r_12*b_12 (1-x_12*b_12) 0 -1 0 0 -x_12 -r_12 0 0; ...
     0 0 -1 0 (1-x_32*b_32) -r_32*b_32 0 0 -r_32 x_32; ... 
     0 0 0 -1 r_32*b_32 (1-x_32*b_32) 0 0 -x_32 -r_32  ...
    ];
b = [real(V1p) imag(V1p) real(V3p) imag(V3p) real(I12p) imag(I12p) ...
     real(I32p) imag(I32p) 0 0 0 0]';

% weighting matrix
W = eye(12,12);
W(5,5) = 1/2.27; W(6,6) = W(5,5); % W(9,9) = W(5,5); W(10,10) = W(5,5);
W(7,7) = 1/5.85; W(8,8) = W(7,7); % W(11,11) = W(7,7); W(12,12) = W(7,7);
W = eye(12,12);
W(9,9) = 100; W(10,10)=W(9,9); W(11,11)=W(9,9); W(12,12)=W(9,9);
% use high weights on ckt eqns
 
% x = (A'*W*A)\(A'*W*b);
x = (W*A)\(W*b);

% compare results
[b(1:8',1), x([1 2 5 6 7 8 9 10],1)];

% raw PMU data
V1 = 206368.1/345000*sqrt(3);
V1ph = -15.912;
V3 = 204659.1/345000*sqrt(3);
V3ph = -24.12;
I_12 = 380.4/167.3479;
I_12ph = 149.856;
I_32 = (978.4/Is)/167.3479;
I_32ph = 159.488;

% LSE results
V1_LSE = sqrt(x(1)^2+x(2)^2);
V1_ph_LSE = atan2(x(2),x(1))*180/pi;
V3_LSE = sqrt(x(5)^2+x(6)^2);
V3_ph_LSE = atan2(x(6),x(5))*180/pi;
I12_LSE = sqrt(x(7)^2+x(8)^2);
I12_ph_LSE = atan2(x(8),x(7))*180/pi;
I32_LSE = sqrt(x(9)^2+x(10)^2);
I32_ph_LSE = atan2(x(10),x(9))*180/pi;
V2_LSE = sqrt(x(3)^2+x(4)^2)
V2_ph_LSE = atan2(x(4),x(3))*180/pi

[V1 V1ph V1_LSE V1_ph_LSE]
[V3 V3ph V3_LSE V3_ph_LSE]
[I_12 I_12ph I12_LSE I12_ph_LSE]
[I_32 I_32ph I32_LSE I32_ph_LSE]

% I_12 calculated from LSE
I_12_PSE = (-V1_LSE*exp(jay*V1_ph_LSE*pi/180) + V2_LSE*exp(jay*V2_ph_LSE*pi/180))/(r_12+jay*x_12)
% 6. Problem 10.14

[num10,txt10,raw10] = xlsread('Prob10-14.xls','A2:D302');
% Channel 1 (column) is time, channel 2 is Malin voltage, channel 3 is COI
% (Cal-Oregon interface), channel 4 is frequency
t = num10(:,1); V_Malin = num10(:,2); P_COI = num10(:,3); f = num10(:,4);
set_font_size
figure, plot(t,P_COI)
xlabel('Time (sec)'), ylabel('Active Power Flow (MW)')
axis([70 85 3400 5000])
reset_font_size

% Remove average flow
P_MW2 = P_COI(:,1) - 4200; % ends at 85 seconds


% the following are the results used in the chapter
n = 100  % 50 and 30 are good; 5 is no good
ts = 0.05;
%figure, plot(P_MW1)
[B,A] = prony(P_MW2, n-1, n);   % MATLAB prony function
syst = tf(B,A,ts);
[pole(syst) abs(pole(syst))]
td2 = 300*0.05;
[y,td] = impulse(syst,td2);
set_font_size
figure, plot(td+70,P_MW2,'k-',td+70,y*ts,'k--')
xlabel('Time (sec)'), ylabel('Active Power Flow (MW)')
legend('Raw data','Prony model')
axis([70 85 -500 600])
reset_font_size

z = 1.0022 + j*0.0682  % pole from n = 2, unstable
s = log(z)/ts
% s = 0.0902 + j*1.3589  ==> 0.2163 Hz, damping ratio = -0.0664 (unstable)
syst_ss = canon(syst,'modal'); % mode is 3,4
syst_small = ss(syst_ss.a(3:4,3:4),syst_ss.b(3:4,1), ...
                syst_ss.c(1,3:4),0,ts)
y2 = impulse(syst_small,td2);
set_font_size
figure, plot(td+70,P_MW2,'k-',td+70,y2*ts,'k--')
xlabel('Time (sec)'), ylabel('Active Power Flow (MW)')
legend('Raw data','Second-order model')
axis([70 85 -500 600])
%% 
% 

reset_font_size
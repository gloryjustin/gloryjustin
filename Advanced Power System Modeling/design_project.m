%% ECSE 6180 Design Project 1
% Glory Justin
% Design a controller to provide damping for power system oscillations.
% 10/26/2020

svm_mgen;
A11 = a_mat([1:8,11:13],[1:8, 11:13]); 
eig(A11)
% from control system model 
A = a_mat;
B = b_dcr;
c1 = (bus_sol(2,2)*sind(bus_sol(3,3)-bus_sol(3,2)))/(0.4/9.91);
c2 = (bus_sol(3,2)*sind(bus_sol(3,3)-bus_sol(3,2)))/(0.4/9.91);
c3 = (bus_sol(3,2)*bus_sol(2,2)*cosd(bus_sol(3,3)-bus_sol(3,2)))/(0.4/9.91);
c4 = -(bus_sol(3,2)*bus_sol(2,2)*cosd(bus_sol(3,3)-bus_sol(3,2)))/(0.4/9.91);
C = c1*c_v(3,:)+c2*c_v(2,:)+c3*c_ang(3,:)+c4*c_ang(2,:);
D = 0;
G = ss(A,B,C,D);

% rlocus(G)

% add washout filter, rate filter, and low pass filter 
WO = tf([10 0],[10 1]);
RF = tf([10 0],[10 1]);
LP = tf([1],[1 20]); 

% put all circuits in series with system
G_f = G*WO*RF*LP

figure, rlocus(G_f)  
%% 
% 


% Damping controller design
% lead_angle = 50;
lead_angle = 60;
wlm = 8.95; % local mode is about 9 rad/sec
alpha = (1+sind(lead_angle))/(1-sind(lead_angle))
z = wlm/sqrt(alpha), p = wlm*sqrt(alpha)
T1 = 1/z, T2 = 1/p
leadlag = tf([T1 1],[T2 1])

% Compensated system 
G_comp = G*leadlag*leadlag*WO*RF*LP

figure, rlocus(G_comp)  
%% 
% 


% select gain as 8 from RL

G_comp_8 = feedback(G_comp*0.1287,1) % note negative feedback
eig(G_comp_8)

% form simulation model - input is exciter voltage reference input

b_vr_18 = [b_vr; 0; 0; 0; 0; 0];
c_p32 = [C,0,0,0,0,0];
G_sim = ss(G_comp_8.a,b_vr_18,G_comp_8.c, 0);
[y,t] = step(0.05*G_sim);
figure, plot(t,y)
xlabel('Time (sec)'), ylabel('\Delta P32 (pu)')
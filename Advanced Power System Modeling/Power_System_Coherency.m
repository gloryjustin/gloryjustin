%% ECSE 6180 HW 6
%% Glory Justin 11/27/2020
% 1. Problem 16.1
% To prove that one is the inverse of the other, we multiply both matrices
% 
% $$\left\lbrack \begin{array}{c}C\\G\end{array}\right\rbrack \left\lbrack \begin{array}{cc}U 
% & G^+ \end{array}\right\rbrack =\left\lbrack \begin{array}{cc}\textrm{CU} & 
% {\textrm{CG}}^+ \\\textrm{GU} & {\textrm{GG}}^+ \end{array}\right\rbrack$$
% 
% Then we analyze each matrix sub component
% 
% $$\textrm{CU}=\left(M_a^{-1} U^T M\right)U=M_a^{-1} \left(U^T \textrm{MU}\right)=M_a^{-1} 
% M_a =I_r$$
% 
% $\textrm{GU}=\textrm{diag}\left(G_1 u_1 ,G_2 u_2 ,\ldotp \ldotp \ldotp ,G_r 
% u_r \right)$ but $G_{\alpha } u_{\alpha } =0$
% 
% Thus $\textrm{GU}=0\;$, a zero matrix
% 
% ${\textrm{GG}}^+ =I_{N_m -r}$ since $G^+$ is the Moore-Penrose pseudoinverse 
% of G
% 
% $$\begin{array}{l}{\textrm{CG}}^+ =\left(M_a^{-1} U^T M\right)\left(M^{-1} 
% G^T {\left({\textrm{GM}}^{-1} G^T \right)}^{-1} \right)=M_a^{-1} U^T \left({\textrm{MM}}^{-1} 
% \right)G^T {\left({\textrm{GM}}^{-1} G^T \right)}^{-1} \\\;\;\;\;\;\;\;\;\;\;=M_a^{-1} 
% {\left(\textrm{GU}\right)}^T {\left({\textrm{GM}}^{-1} G^T \right)}^{-1} \;\textrm{but}\;\textrm{GU}=0\;\textrm{so}\;{\textrm{CG}}^+ 
% =0\end{array}$$
% 
% Thus $\left\lbrack \begin{array}{c}C\\G\end{array}\right\rbrack \left\lbrack 
% \begin{array}{cc}U & G^+ \end{array}\right\rbrack =\left\lbrack \begin{array}{cc}I_r  
% & 0\\0 & I_{N_m -r} \end{array}\right\rbrack$. SInce $\left\lbrack \begin{array}{c}C\\G\end{array}\right\rbrack 
% \left\lbrack \begin{array}{cc}U & G^+ \end{array}\right\rbrack$ is an identity 
% matrix, $\left\lbrack \begin{array}{cc}U & G^+ \end{array}\right\rbrack \;\textrm{is}\;\textrm{the}\;\textrm{inverse}\;\textrm{of}\;\left\lbrack 
% \begin{array}{c}C\\G\end{array}\right\rbrack$
% 2. Problem 16.2
% $$\begin{array}{l}M\ddot{x} =\textrm{Kx}\;\;\textrm{thus}\;\ddot{x} =M^{-1} 
% \textrm{Kx}\\\left\lbrack \begin{array}{c}\ddot{y} \\\ddot{z} \end{array}\right\rbrack 
% =\left\lbrack \begin{array}{c}C\\G\end{array}\right\rbrack \ddot{x} =\left\lbrack 
% \begin{array}{c}C\\G\end{array}\right\rbrack M^{-1} K\left(U\;G^+ \right)\left\lbrack 
% \begin{array}{c}y\\z\end{array}\right\rbrack \;\\\ddot{y\;} ={\textrm{CM}}^{-1} 
% K\left(\textrm{Uy}+G^+ z\right)\;\;\textrm{but}\;C=M_a^{-1} U^T M\;\textrm{thus}\\\ddot{y} 
% =M_a^{-1} U^T {M\;M}^{-1} K\;U\;y+M_a^{-1} U^T M\;M^{-1} {K\;G}^+ z\;\\\textrm{Multiplying}\;\textrm{by}\;M_{a\;} 
% \;\textrm{gives}\\M_a \ddot{y} =U^T \textrm{KUy}+U^T {\textrm{KG}}^T {\left({\textrm{GG}}^T 
% \right)}^{-1} z\;\\\textrm{but}\;K=K^I +\varepsilon K^E \;,\textrm{substituting}\;\textrm{for}\;K\;\textrm{and}\;\textrm{neglecting}\;K^{I\;} 
% \;\textrm{yields}\\M_a \ddot{y} =\varepsilon U^T K^E {\textrm{Uy}} +\varepsilon 
% U^T K^E M^{-1} G^T {\left({{\textrm{GM}}^{-1} G}^T \right)}^{-1} z\;\\\;\;\;\;\;\;\;\;=\varepsilon 
% K_a y+\varepsilon K_{\textrm{ad}} z\;\textrm{where}\;K_a =U^T K^E U\;\textrm{and}\;K_{\textrm{ad}} 
% =U^T K^E M^{-1} G^T M_d \\\ddot{z} ={\textrm{GM}}^{-1} K\left(\textrm{Uy}+G^+ 
% z\right)={\textrm{GM}}^{-1} \textrm{KUy}+{\textrm{GM}}^{-1} {\textrm{KG}}^+ 
% z\\\;\;\;={\textrm{GM}}^{-1} \textrm{KUy}+{\textrm{GM}}^{-1} {\textrm{KG}}^T 
% {\left({\textrm{GG}}^T \right)}^{-1} z\\\textrm{Multiplying}\;\textrm{by}\;M_d 
% \;\textrm{yields}\\M_d \ddot{z} =M_d {\textrm{GM}}^{-1} \textrm{KUy}+M_d {\textrm{GM}}^{-1} 
% {\textrm{KG}}^T {\left({\textrm{GG}}^T \right)}^{-1} z\\\textrm{But}\;K=K^I 
% +\varepsilon K^E ,\;\textrm{substituting}\;\textrm{for}\;K\;\textrm{and}\;\textrm{neglecting}\;K^I 
% \;\textrm{for}\;\textrm{the}\;y\;\textrm{variable}\;\\M_d \ddot{z} =M_d {\textrm{GM}}^{-1} 
% K^E \textrm{Uy}+M_d {\textrm{GM}}^{-1} {\left(K^I +{\varepsilon K}^E \right)G}^T 
% {\left({\textrm{GG}}^T \right)}^{-1} z\\G^T {\left({\textrm{GG}}^T \right)}^{-1} 
% =M^{-1} G^T {\left({\textrm{GM}}^{-1} G^T \right)}^{-1} =M^{-1} G^T M_d \\\textrm{Thus}\;M_d 
% \ddot{z} =\varepsilon K_{\textrm{da}} y+\left(K_d +\varepsilon K_{\textrm{dd}} 
% \right)z\;\textrm{where}\;K_{\textrm{da}} =M_d {\textrm{GM}}^{-1} K^E U,K_d 
% =M_d {\textrm{GM}}^{-1} K^I M^{-1} G^T M_d \;\textrm{and}\;K_{\textrm{dd}} =M_d 
% {\textrm{GM}}^{-1} K^E M^{-1} G^T M_d \end{array}$$
% 3. Problem 16.3
% Using the code from Example 16.2, first we find the eigenvalues of the A matrix 
% and the stiffness matrix K.

svm_mgen  % select  d2a_em_coh.m   in dialog box
a_mat
eig(a_mat)

irow = [2 4 6 8]; icol = [1 3 5 7];
MinvK = a_mat(irow,icol) 
[V D] = eig(MinvK*377)

% stiffness matrix K = 2*M*MinvK
 K = 2*6.5*diag([9 9 11 7])*MinvK

% modal frequency
% modes = sqrt(abs(diag(D))*377);   % in rads/sec
%% 
% The interarea mode is sqrt(-13.7579)= +/- 3.709 rad/s while the local modes 
% are sqrt(-60.5561)= +/- 7.7818 rad/s and sqrt(-63.181)= +/- 7.949 rad/s. To 
% decompose K into the internal and external connection matrixes


% constructing reduced models
KI = zeros(4,4);
KI(1,2) = K(1,2); KI(1,1) = -KI(1,2);
KI(2,1) = K(2,1); KI(2,2) = -KI(2,1);
KI(3,4) = K(3,4); KI(3,3) = -KI(3,4);
KI(4,3) = K(4,3); KI(4,4) = -KI(4,3);
KI
KE = K - KI

H1 = 9*6.5; H2 = 9*6.5; H3 = 7*6.5; H4 = 11*6.5;   
M = 2*diag([H1,H2,H3,H4])
Ma = 2*diag([H1+H2,H3+H4])
U = [1 0; 1 0; 0 1; 0 1];
Ka = U'*KE*U
G = [-1 1 0 0; 0 0 -1 1];
Gp = inv(M)*G'*inv(G*inv(M)*G');
A11 = inv(Ma)*U'*KE*U
A12 = inv(Ma)*U'*KE*Gp
A21 = G*inv(M)*KE*U
A22 = G*inv(M)*KI*Gp
A22p = G*inv(M)*KE*Gp
Md = inv(G*inv(M)*G')
Kd = Md*G*inv(M)*KI*inv(M)*G'*Md

eig(A11*377)
aggreg_mode = sqrt(abs(eig(A11)))

eig(A22)
local_mode = sqrt(abs(eig(A22))*377)

A11_corr = A11 - A12*inv(A22+A22p)*A21 

eig(A11_corr)
aggreg_mode = sqrt(abs(eig(A11_corr))*377)

A11_corr1 = Ma*A11_corr
% 4. Problem 16.4
% To find the coherent areas, there are 2 coherent areas with one interarea 
% mode of frequency 3.709 rad/s

[Mfull, Dfull] = eig(MinvK)
Vs = Mfull(:,1:2);
Vsx = Vs;
for i=2:4
    Vsx(i,:) = Vs(i,:) - Vs(1,:);
end
Vsx
Vsord = [Vs(1,:); Vs(3,:); Vs(2,:); Vs(4,:)]
I_L = Vsord * inv(Vsord(1:2,:))

c_angle = c_ang(:,[1 3 5 7])

V_bus_angle = c_angle*Vs

I_LV = V_bus_angle * inv(Vsord(1:2,:))
%% 
% From I_LV above, the system can be grouped into 2 coherent areas
% 
% Area 1- generators: 1, 2 buses: 1, 2, 3, 10, 20
% 
% Area 2- generators: 11, 12 buses: 11, 12, 13, 110, 120
% 5. Problem 16.5
% The parameters are

jay = sqrt(-1); theta_11 = 0; theta_12 = -16.6;
V_11 = 1.03*exp(jay*0); V_12 = 1.05*exp(-jay*16.6*pi/180);
S_11 = 9.2471e+000 + jay*1.2000e+000; 
S_12 = 5.0000e+000 + jay*3.2249e+000;
xdpp_11 = 0.25/11; xdpp_12 = 0.25/7;
%% 
% For the podmore aggregation method

Veq = (real(S_11)*V_11 + real(S_12)*V_12)/(real(S_11)+real(S_12))
theta_eq = (real(S_11)*theta_11 + real(S_12)*theta_12)/(real(S_11)+real(S_12))
%% 
% Using the inertial aggregation method

I_11 = conj(S_11/V_11);
I_12 = conj(S_12/V_12);
Ep_11 = V_11 + jay*xdpp_11*I_11;
Ep_12 = V_12 + jay*xdpp_12*I_12;
abs(Ep_11), angle(Ep_11)*180/pi
abs(Ep_12), angle(Ep_12)*180/pi
V_cm_mag = (real(S_11)*Ep_11 + real(S_12)*Ep_12)/(real(S_11)+real(S_12))
V_cm_ph  = (real(S_11)*theta_11 + real(S_12)*theta_12)/(real(S_11)+real(S_12))/2*180/pi
V_cm     = V_cm_mag*exp(jay*V_cm_ph*pi/180)
phi_11 = angle(Ep_11)*180/pi - V_cm_ph
phi_12 = angle(Ep_12)*180/pi - V_cm_ph
t_11   = V_cm_mag/abs(Ep_11)
t_12   = V_cm_mag/abs(Ep_12)
I_eq = I_11 * exp(-jay*phi_11*pi/180)/t_11 + ...
       I_12 * exp(-jay*phi_12*pi/180)/t_12
S_cm = V_cm*conj(I_eq)
V_eq = V_cm - jay*I_eq*0.25/18
abs(V_eq), angle(V_eq)*180/pi
% 6. Problem 16.6

jay = sqrt(-1);
Y14 = -1/(jay*0.0167);
Y24 = -1/(0.001 + jay*0.01);
Y34 = -1/(0.0025 + jay*0.025);
Y11 = -Y14;
Y22 = -Y24 + jay*0.0175/2;
Y33 = -Y34 + jay*0.0437/2;
Y44 = Y11 + Y22 + Y33 + jay*1;
Y = [ Y11  0    0    Y14; ...
      0    Y22  0    Y24; ...
      0    0    Y33  Y34; ...
      Y14  Y24  Y34  Y44]

Ya = [Y11  0    0    ; ...
      0    Y22  0    ; ...
      0    0    Y33 ];
Yb = [Y14; Y24; Y34]; 
Yc = [Y14 Y24 Y34];
Yred = Ya - Yb*inv(Y44)*Yc
% get individual line reactance
Line12_13  = -1/Yred(1,2) 
Line12_110 = -1/Yred(1,3) 
Line13_110 = -1/Yred(2,3) 
Bus_load = sum(Yred) 
% 7. Problem 16.7

jay = sqrt(-1);
Y16 = -1/(0.0035+jay*0.0411);
Y26 = -1/(0.032 + jay*0.032);
Y36 = -1/(0.0008 + jay*0.0074);
Y46 = -1/(0.0016+ jay*0.0163);
Y56 = -1/(0.0013 + jay*0.0188);
Y11 = -Y16 + jay*0.6987/2;
Y22 = -Y26 + jay*0.41/2;
Y33 = -Y36 + jay*0.48/2;
Y44 = -Y46 + jay*0.25/2;
Y55 = -Y56 + jay*1.31/2;
Y66 = Y11 + Y22 + Y33 + Y44 + Y55;
Y = [ Y11  0    0   0  0  Y16; ...
      0    Y22  0   0  0  Y26; ...
      0    0    Y33 0  0  Y36; ...
      0    0    0   Y44  0  Y46; ...
      0    0    0   0  Y55  Y56; ...
      Y16  Y26  Y36  Y46  Y56 Y66]

Ya = [Y11  0    0   0   0 ; ...
      0    Y22  0   0   0 ; ...
      0    0    Y33 0   0 ; ...
      0    0    0   Y44 0 ; ...
      0    0    0   0   Y55];
Yb = [Y16; Y26; Y36; Y46; Y56]; 
Yc = [Y16 Y26 Y36 Y46 Y56];
Yred = Ya - Yb*inv(Y66)*Yc

Line2_27  = -1/Yred(1,2) 
Line2_30 = -1/Yred(1,3) 
Line2_31 = -1/Yred(1,4) 
Line2_47 = -1/Yred(1,5)
Line27_30 = -1/Yred(2,3)
Line27_31 = -1/Yred(2,4)
Line27_47 = -1/Yred(2,5)
Line30_31 = -1/Yred(3,4)
Line30_47 = -1/Yred(3,5)
Line31_47 = -1/Yred(4,5) 
Bus_load = sum(Yred)
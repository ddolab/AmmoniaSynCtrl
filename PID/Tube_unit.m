  function [dx, y] = Tube_unit(t, x, u, d_o,len, varargin)
  %A single unit of ammonia tube reactor.
  %Parameters
    NoComp  = 3; %[H2 N2 NH3]
    MW_comp = [2 28.01 17.03]*1e-3; %molecular weight
    R       = 8.314;
    D_p     = .00285; %catalyst particle diameter
    cp_c    = 1100; %catalyst heat capacity J/kg/K
    rho_c   = 2200; %catalyst denstiy 2200 kg/m3
    eps      = .3; %bed void fraction
    mu_g_i  = 0.028; %cP viscosity
    nambda_g_i = 0.231 ;%W/mK thermal conductivity
    
  %Variables
%     len         AS LENGTH_M_RAC
%     vol         AS VOLUME_M_RAC
%     d_o         AS LENGTH_L_RAC
%     d_i         AS LENGTH_L_RAC
%     area        AS AREA_M_RAC

%     F_g         AS  MOLEFLOWRATE_M_RAC
%     z_g         AS  ARRAY(NoComp) OF MOLEFRAC_RAC
%     T_g         AS  TEMPERATUREK_H_RAC
%     T_c         AS  TEMPERATUREK_H_RAC
%     z_mass      AS  MASSFRAC_RAC
%     p_g_p       AS  ARRAY(NoComp) PRESSURE_M_RAC
%     k1          AS  NoType_pos_L_RAC
%     k2          AS  NoType_pos_M_RAC
%     enth_rac    AS  MASSENTHALPY_M_RAC
%     cp_g        AS  MOLEHEATCAPACITY_M_RAC
%     rho_g       AS  MOLEDENSITY_H_RAC
%     r_rac       AS  REACTIONRATE_M_RAC
%     MW_g        AS  MOLEMASS_M_RAC
% 
%     Re_g        AS  NoType_pos_H_RAC
%     Pr_g        AS  NoType_pos_LM_RAC
%     U_gc        AS  VOLHEATTRANSFER_M_RAC
% 
%     F_g_i       AS MOLEFLOWRATE_M_RAC
%     F_g_o       AS MOLEFLOWRATE_M_RAC
%     z_g_i       AS ARRAY(NoComp) OF MOLEFRAC_RAC
%     z_g_o       AS ARRAY(NoComp) OF MOLEFRAC_RAC
%     T_g_i       AS TEMPERATUREK_H_RAC
%     T_g_o       AS TEMPERATUREK_H_RAC
%     P_g_i       AS PRESSURE_H_RAC
%     P_g_o       AS PRESSURE_H_RAC
%     z_mass_i    AS MASSFRAC_RAC
%     m_g_i       AS MASSFLOWRATE_M_RAC
%     MW_g_i      AS MOLEMASS_M_RAC
%     v_g_i       AS VELOCITY_M_RAC
%     rho_g_i     AS MOLEDENSITY_H_RAC

  %Input structure
     F_g_i = u(1);
     T_g_i = u(2);
     P_g_i = u(3);
     z_g_i = u(4:6);
  
  %dimension
  d_i = d_o; 
  area = d_i^2*pi/4;
  vol = d_o^2*pi/4*len;
  
  %input properties
  MW_g_i = dot(MW_comp,z_g_i);
  m_g_i = F_g_i*MW_g_i;
  rho_g_i = P_g_i*1e5/R/T_g_i; % assume ideal gas
  v_g_i = F_g_i/rho_g_i/area;
  z_mass_i = z_g_i(3)*MW_comp(3)/MW_g_i;
  
  %state structure
  T_g_o = x(1);
  T_c = x(2);
  
  k1 = 1.79e4 * exp(-87090/(R*T_c));
  k2 = 2.57e16 * exp(-198464/(R*T_c));
  P_g_o = P_g_i - m_g_i/(rho_g_i*MW_g_i)/area/D_p * (1-eps)/eps^3*(150*(1-eps)*(mu_g_i*1e-3)/D_p + 1.75*m_g_i/area)*len/1e5;
  
  %Mass balance
  x0 = [F_g_i z_g_i];
  %options = optimoptions(@fsolve,'Algorithm','levenberg-marquardt');
  options = optimset('Display','off');
  mb = zeros(1,4);
  if (F_g_i>=0)&&(T_g_i>0)&&(P_g_i>0)&&(z_g_i(1)>=0)&&(z_g_i(2)>=0)&&(z_g_i(3)>=0)
      mb = fsolve(@(s) mass_balance(s, P_g_i, MW_comp, k1, k2, rho_c, z_mass_i, m_g_i, eps, area, F_g_i, z_g_i, len), x0, options);
  end
  %disp(u)
  F_g_o = mb(1);
  z_g_o = mb(2:4);
  %output properties
  MW_g = dot(MW_comp,z_g_o);
  m_g = F_g_o*MW_g;
  rho_g = P_g_o*1e5/R/T_g_o;
  v_g = F_g_o/rho_g/area;
  z_mass = z_g_o(3)*MW_comp(3)/MW_g;
  
  p_g_p = P_g_i*z_g_o;
  enth_rac = -4.184/MW_comp(3)*(-9184 - 7.2949*T_g_o + 0.34996e-2*(T_g_o)^2 + 0.03356e-5*(T_g_o)^3 - 0.11625e-9*(T_g_o)^4 - (6329.3 - 3.1619*(P_g_i*.98692)) + (14.3595 + 4.4552e-3*(P_g_i*0.98692))*(T_g_o) - (T_g_o)^2*(8.3395e-3 + 1.928e-6*P_g_o*0.98692) - 51.21 + 0.14215*P_g_o*0.98692)*1e-3;
  r_rac = (k1*p_g_p(2)*(p_g_p(1)+1e-5)^1.5/(p_g_p(3)+1e-5) - k2*p_g_p(3)/(p_g_p(1)+1e-5)^1.5)*34/rho_c*4.75; % Ref [Morud and Skogestad 1998] [kg NH3/kg cat/hr]

  %Heat transfer
  CP = Cp(T_g_o, P_g_o, z_g_o);
  Pr_g = CP/MW_g*(mu_g_i*1e-3)/nambda_g_i+.001;
  Re_g = D_p*(m_g_i/area)/(1-eps)/(mu_g_i*1e-3)+1;
  U_gc = nambda_g_i/(pi*D_p^2)*(2+1.1*(Re_g+1e-5)^.6*(Pr_g+1e-5)^.33)*1e-3;

  % State equations.
  if (F_g_i>0)&&(T_g_i>0)&&(P_g_i>0)&&(z_g_i(1)>=0)&&(z_g_i(2)>=0)&&(z_g_i(3)>=0)
        dx = [- m_g_i/MW_g/eps/rho_g/area*((T_g_o - T_g_i)/len) + U_gc*1e3/eps/rho_g/CP*(T_c-T_g_o) ...
        enth_rac*1e3*(r_rac/3600)/cp_c - U_gc*1e3/(1-eps)/rho_c/cp_c*(T_c-T_g_o)];
        y = [F_g_o T_g_o P_g_o z_g_o];
  else
      dx = [0 0];
      y = u;
  end
  % Output equations.
  
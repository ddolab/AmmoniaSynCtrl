function M = mass_balance(x, p_g_i, MW_comp, k1, k2, rho_c, z_mass_i, m_g_i, eps, area, F_g_i, z_g_i, len)

% Mass balance of the reactor
% x(1): Molar Flowrate
% x(2)~x(4): Molar fractions
MW_g = dot(MW_comp,x(2:4));
z_mass = x(4)*MW_comp(3)/MW_g;
p_g_p = p_g_i*x(2:4);
r_rac = (k1*p_g_p(2)*(p_g_p(1)+1e-5)^1.5/(p_g_p(3)+1e-5) - k2*p_g_p(3)/(p_g_p(1)+1e-5)^1.5)*34/rho_c*4.75; % Ref [Morud and Skogestad 1998] [kg NH3/kg cat/hr]
M(1) = - m_g_i/eps/area*((z_mass - z_mass_i)/(len)) + (1-eps)*rho_c/eps*(r_rac/3600);
M(2) = x(2) + x(3) + x(4) - 1;
M(3) = x(1)*x(3) - F_g_i*z_g_i(2) + (x(1)*x(4) - F_g_i*z_g_i(3))/2;
M(4) = x(1)*x(2) - F_g_i*z_g_i(1) + (x(1)*x(4) - F_g_i*z_g_i(3))/2*3;
%M(5) = x(1)*x(4) - m_g_i*z_mass/MW_comp(3);
end


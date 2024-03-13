function F = VLE(X, T_sep, P_sep, N_sep)
F = zeros(8,1);
K = 1/P_sep*10^(4.8688-1113.928/(T_sep-10.409));
Henry_const = [-3.68607  -2.29337;
               0.596736*1e4 0.5294740*1e4;
               -0.642828*1e6,-0.521881*1e6];
T_array = [1;1/T_sep;1/T_sep^2];
H = exp(Henry_const'*T_array);
F(1) = X(1)*X(3) + X(2)*X(6) - N_sep(1)/1e6;
F(2) = X(1)*X(4) + X(2)*X(7) - N_sep(2)/1e6;
F(3) = X(1)*X(5) + X(2)*X(8) - N_sep(3)/1e6;
F(4) = X(3)*P_sep/1.013 - H(1)*X(6);
F(5) = X(4)*P_sep/1.013 - H(2)*X(7);
F(6) = X(5) - K*X(8);
F(7) = X(3) + X(4) + X(5) - 1;
F(8) = X(6) + X(7) + X(8) - 1;
end
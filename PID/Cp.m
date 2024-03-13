function CP = Cp(T,P,z)
%Calculate the heat capacity for the reactor
    C = zeros(3,7);
    CP_comp = zeros(1,3);
    C(1,:) = [33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974];%H2
    C(2,:) = [19.50583, 19.88705, -8.598535, 1.369784, .527601, -4.935202, 212.39];%N2
    C(3,:) = [19.99563, 49.77119, -15.37599, 1.921168, .189174, -53.30667, -45.89806];%NH3
    for i = 1:2
        CP_comp(i) = (C(i,1) + C(i,2)*(T/1000) + C(i,3)*(T/1000)^2 + C(i,4)*(T/1000)^3 + C(i,5)/(T/1000)^2);
    end
    CP_comp(3) = 4.184*(6.5846 - 0.61251e-2*T + 0.23663e-5*T^2 - 1.5981e-9*T^3 + (96.1678-0.067571*P*0.98692) + (-.2225 + 1.6847e-4*P*0.98692)*T + (1.289e-4 - 1.0095e-7*P*.98692)*T^2);
    CP = dot(z,CP_comp);
end


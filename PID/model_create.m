function nlgr = model_create(Tc, Tg, l, d)
FileName      = 'Tube_unit';                          % File describing the model structure.
Order         = [6 6 2];                           % Model orders [ny nu nx].
Parameters    = [d;l];         % Initial parameters.
InitialStates = [Tc;Tg];                 % Initial value of the initial states.
Ts            = 0;                                 % Time-continuous system.
nlgr = idnlgrey(FileName, Order, Parameters, InitialStates, Ts, 'Name', ...
                'Ammonia reactor',  ...
                'TimeUnit', 'seconds');
nlgr.InputName = {'Molar Flowrate' ...   % u(1).
                  'Inlet feed stream temperature' ...    % u(2).
                  'Feed Pressure' ...  % u(3).
                  'Molar Fraction of H2' ... %u(4).
                  'Molar Fraction of N2' ... %u(5).
                  'Molar Fraction of NH3'}; %u(6).
%nlgr.InputUnit = {'kgmol/m^3' 'K' 'K'};
nlgr = setinit(nlgr, 'Name', {'Temperature of catalyst' ...   % x(1).
                       'Reactor Temperature'});             % x(2).
nlgr = setinit(nlgr, 'Unit', {'K' 'K'});
nlgr = setinit(nlgr, 'Fixed', {false false});
nlgr.OutputName = {'Molar Flowrate' ...   % y(1).
                  'Outlet stream temperature'  ...      % y(2).
                  'Outlet Pressure' ...  % y(3).
                  'Molar Fraction of H2' ... %y(4).
                  'Molar Fraction of N2' ... %y(5).
                  'Molar Fraction of NH3'}; %y(6).
nlgr = setpar(nlgr, 'Name', {'Diameter'...  % d_o.
                             'Length'}); %len
nlgr.Parameters(1).Fixed = true;   % Fix d_o.
nlgr.Parameters(2).Fixed = true;   % Fix len.
nlgr.Parameters(1).Minimum = 0.1;
nlgr.Parameters(2).Minimum = 0.01;
%present(nlgr);
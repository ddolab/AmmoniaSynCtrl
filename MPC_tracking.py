from math import pi,exp
from pyomo.environ import *
from pyomo.dae import *
import numpy as np
import gym
from scipy.optimize import fsolve
from scipy.integrate import odeint,solve_ivp
import matplotlib
import matplotlib.pyplot as plt
from reactor_env import NH3reactor
from OptSteady_mass import Optsteady
from feasiblity import prediction_feasibility
from ML_prediction import ML_predict
from MovingHorizonEst import MHP
import csv
import time
import pickle
import cloudpickle

def MPC_loop(workers_index, N = 10):
    dataset_size = 0
    while True:
        try:
            size = MPC_running(workers_index)
        except:
            pass
        
    return dataset_size

def MPC_running(index, _seed = None):
    data_set = []
    file_name = 'dataset_' + str(index)
    with open('dynamic_opt_stable', mode='rb') as file:
        opt_result = pickle.load(file)
    total_time_horizon = 3600*5
    time_prediction = 180
    time_action = 10
    time_implementation = 30
    time_interval_schedule = 60
    num_steps_prediction = int(time_prediction/time_action)
    num_steps_implementation = int(time_implementation/time_action)
    num_steps_ttl = int(total_time_horizon/time_action)
    enable_recycle = True
    feed_profile = []
    num_feed = 5
    rng = np.random.default_rng(seed =_seed)
    for i in range(num_feed):
        if i == 0:
            feed_profile = np.append(feed_profile, rng.integers(low = 300, high = 600)*np.ones(int(num_steps_ttl/num_feed)))
        else:
            feed_profile = np.append(feed_profile, rng.integers(low = 300, high = 600)*np.ones(int(num_steps_ttl/num_feed)))
    #feed_profile = 550*np.ones(int(num_steps_ttl/4))
    #feed_profile = np.append(feed_profile, 600*np.ones(int(num_steps_ttl/4*3)))
    feed_profile = np.append(feed_profile, feed_profile[-1]*np.ones(num_steps_prediction))
    print(feed_profile)
    epsilon = 1 #nominal conversion
    product_profile = feed_profile/2*epsilon
    mykeys = list(opt_result['mp'].keys())
    mykeys.sort()
    #product_profile = np.array(list({i: opt_result['mp'][i] for i in mykeys}.values()))
    P_profile = np.array(list({i: opt_result['P'][i] for i in mykeys}.values()))
    Nl_profile = np.array(list({i: opt_result['Nl'][i] for i in mykeys}.values()))
    product_profile = np.append(product_profile, product_profile[-1]*np.ones(num_steps_prediction))
    P_profile = np.append(P_profile, P_profile[-1]*np.ones(num_steps_prediction))
    Nl_profile = np.append(Nl_profile, Nl_profile[-1]*np.ones(num_steps_prediction))
    Ft = feed_profile[0]
    #design parameters
    L = np.array([1.24,1.79,2.83]) #length, m
    D = 1.2 #diameter, m
    RR = 5 #recycle ratio
    T_sep = 313 #separation temperature in the flash tank
    P_sep = 196 #separation pressure
    V_sep = 100 #volume of the flash tank, m3
    HX_A = 180
    HX_U = 536
    Henry_const = np.array([[-3.68607, -2.29337],
                        [0.596736*1e4, 0.5294740*1e4],
                        [-0.642828*1e6,-0.521881*1e6]])
    H = np.exp(np.matmul(Henry_const.transpose(),np.array([1,1/T_sep,1/T_sep**2])))
    liquid_frac = 0.3 #nominal liquid fraction in the flash tank
    Tt = 523 #input temperature
    P = 200 #pressure, bar
    zt =  np.array([0.75,0.25,0]) #input molar fractions
    MW_comp = np.array([2, 28.01, 17.03])*1e-3 #molecular weight
    molar_weight = sum(MW_comp*zt)

    def prediction(Tg0,Tc0,P0,Nl0,z_NH3_prod,
                   _feed_profile,_product_profile,_P_profile, _Nl_profile,
                   dn_model = None, st_model = None, _enable_recycle = True,time = 3000,interval=10, 
                _init_action = None, _next_action = None):

        CP_comp = np.array([14.6,1.1,3.16])*1000
        NoComp  = range(3) #[H2 N2 NH3]
        z_mass_t = zt*MW_comp/molar_weight #input mass fractions
        l = L/10
        A = D**2/4*pi
        R       = 8.314
        D_p     = .00285 #catalyst particle diameter
        cp_c    = 1100 #catalyst heat capacity J/kg/K
        rho_c   = 2200 #catalyst denstiy 2200 kg/m3
        eps      = .33 #bed void fraction
        mu_g_i  = 0.028 #cP viscosity
        nambda_g_i = 0.231 #W/mK thermal conductivity
        CPt = np.dot(CP_comp,z_mass_t)
        K = 1/P_sep*10**(4.8688-1113.928/(T_sep-10.409)) #separation constant for ammonia
        model = ConcreteModel()
        model.nbed = RangeSet(0,2)
        model.valves = RangeSet(0,3)
        model.seg = RangeSet(1,10)
        model.seg_sub = RangeSet(2,10)
        model.t = ContinuousSet(bounds = (0,time))
        model.Fin_ref = Var(model.t, within=PositiveReals) #feed flowrate reference, kg/s
        model.Fout_ref = Var(model.t, within=PositiveReals) #production rate reference, mol/s
        model.P_ref = Var(model.t, within=PositiveReals) #pressure reference, bar
        model.Nl_ref = Var(model.t, within=PositiveReals) #liquid volume reference, mol
        model.mt = Var(model.t, bounds = (0,100))
        model.Fp = Var(model.t, bounds = (0,500))
        model.Tbed_0 = Var(model.nbed,model.t, bounds = (500,800),initialize=700)
        model.z_mass_bed_0 = Var(NoComp, model.nbed, model.t, bounds=(0,1),initialize=0.33)
        model.sp_r = Var(model.valves,model.t,bounds=(0,1))
        model.Tin = Param(model.t,default=Tt)
        model.z_mass_in = Var(NoComp,model.t,bounds=(0,1))
        #model.CPin = Var(model.t,bounds=(0,100000))
        model.Tg = Var(model.nbed,model.seg,model.t,bounds=(500,900))
        model.Tc = Var(model.nbed,model.seg,model.t,bounds=(500,900))
        model.z_mass = Var(NoComp,model.nbed,model.seg,model.t,bounds=(0,1))
        model.p = Var(NoComp,model.nbed,model.seg,model.t,bounds=(0,200))
        model.CP = Var(model.nbed,model.seg,model.t,bounds=(0,20000))
        model.r_rac = Var(model.nbed,model.seg,model.t,bounds=(-10000,10000))
        model.m = Var(model.nbed,model.t,bounds=(0,200))
        
        if _enable_recycle == True:
            model.mcyc = Var(model.t,bounds=(0,200))
            model.frac_purge = Var(model.t,bounds=(0,0.05), initialize = 0.0001)
            model.Fcyc = Var(model.t,bounds=(0.1*600,RR*600))
            model.z_prod = Var(NoComp,model.t,bounds=(0,1))
            model.z_cyc = Var(NoComp,model.t,bounds=(0,1))
            model.N_sep = Var(NoComp,model.t,within=NonNegativeReals)
            model.Nl_sep = Var(model.t,bounds=(0,None), initialize = V_sep*liquid_frac*40080)
            model.Nv_sep = Var(model.t,bounds=(0,None), initialize =P_sep*1e5*V_sep*(1-liquid_frac)/8.314/T_sep)
            model.dN =  DerivativeVar(model.N_sep, wrt=model.t)
            model.P_sep = Var(model.t, bounds=(0,None), initialize = P_sep)
            #model.P_int = Var(model.t,initialize = 0)
            #model.P_diff = DerivativeVar(model.P_int, wrt=model.t)
        else:
            model.mcyc = Param(model.t,default=0)
            model.z_mass_cyc = Param(NoComp,model.t,default=1/3)

        model.dTcdt = DerivativeVar(model.Tc, wrt=model.t)
        model.dTgdt = DerivativeVar(model.Tg, wrt=model.t)
        model.scale = Param(initialize=1E-1)
        model.T_1 = Var(model.t, bounds=(500,800)) #inlet temperature to the first bed

        #constraints
        def HeatCP_rule(model,n,s,t):

            return model.CP[n,s,t] == sum(CP_comp[i]*model.z_mass[i,n,s,t] for i in NoComp)
        model.HeatCP = Constraint(model.nbed,model.seg,model.t,rule=HeatCP_rule)
        '''
        def HeatCPin_rule(model,t):

            return model.CPin[t] == sum(CP_comp[i]*model.z_mass_in[i,t] for i in NoComp)
        model.HeatCPin = Constraint(model.t,rule=HeatCPin_rule)
        '''
        def F_total_rule(model,t):
            return sum(model.sp_r[n,t] for n in model.valves) == 1
        model.F_total = Constraint(model.t,rule=F_total_rule)

        def T_total_rule(model,t):

            return model.mt[t]*Tt*CPt + model.mcyc[t]*T_sep*sum(CP_comp[i]*model.z_mass_cyc[i,t] for i in NoComp)== (model.mt[t]+model.mcyc[t])*model.Tin[t]*sum(CP_comp[i]*model.z_mass_in[i,t] for i in NoComp)
        #model.T_total = Constraint(model.t,rule=T_total_rule)

        def z_total_rule(model,i,t):
            
            return model.z_mass_in[i,t]*(model.mcyc[t] + model.mt[t]) == model.z_cyc[i,t]*model.Fcyc[t]*MW_comp[i] + model.mt[t]*z_mass_t[i]
        model.z_total = Constraint(NoComp,model.t,rule=z_total_rule)

        def F_in_rule(model,n,t):
            if n > 0:
                return model.m[n,t] == model.sp_r[n+1,t]*(model.mt[t] + model.mcyc[t]) + model.m[n-1,t]
            else:
                return model.m[n,t] == (model.sp_r[0,t]+model.sp_r[1,t])*(model.mt[t] + model.mcyc[t])
        model.F_in = Constraint(model.nbed,model.t,rule=F_in_rule)

        def z_in_rule(model,i,n,t):
            if n > 0:
                return model.z_mass_bed_0[i,n,t]* model.m[n,t] == model.z_mass_in[i,t]*model.sp_r[n+1,t]*(model.mt[t] + model.mcyc[t]) + model.z_mass[i,n-1,10,t]*model.m[n-1,t]
            else:
                return model.z_mass_bed_0[i,n,t] == model.z_mass_in[i,t]
        model.z_in = Constraint(NoComp,model.nbed,model.t,rule=z_in_rule)

        def T_1_rule(model,t):

            Fcold = model.sp_r[0,t]*(model.mt[t] + model.mcyc[t])
            #Fhot = model.m[2,t]
            Cmin = Fcold*sum(CP_comp[i]*model.z_mass_in[i,t] for i in NoComp)
            #Cmax = Fhot*model.CP[2,10,t]
            #Cr = Cmin/Cmax
            NTU = HX_U*HX_A/Cmin
            #epsilon = (1-exp(-NTU*(1-Cr)))/(1-Cr*exp(-NTU*(1-Cr)))
            epsilon = NTU/(1+NTU)
            return model.T_1[t] == epsilon*(model.Tg[2,10,t]-model.Tin[t]) + model.Tin[t]
        model.T_1_eq = Constraint(model.t,rule=T_1_rule)
        
        def delta_T_rule(model,t):
            return model.Tg[2,10,t] >= model.Tin[t] + 20
        model.delta_T = Constraint(model.t,rule=delta_T_rule)

        def T_in_rule(model,n,t):
            if n > 0:
                return model.sp_r[n+1,t]*(model.mt[t] + model.mcyc[t])*sum(CP_comp[i]*model.z_mass_in[i,t] for i in NoComp)*model.Tin[t] + model.m[n-1,t]*model.CP[n-1,10,t]*model.Tg[n-1,10,t] == model.m[n,t]*model.Tbed_0[n,t]*model.CP[n,1,t]
                #return model.sp_r[n+1,t]*(Ft + model.Fcyc[t])*model.Tin[t] + model.F[n-1,10,t]*model.Tg[n-1,10,t] == model.Fbed_0[n,t]*model.Tbed_0[n,t]
            else:
                return model.T_1[t]*model.sp_r[0,t] + model.Tin[t]*model.sp_r[1,t] == model.Tbed_0[n,t]*(model.sp_r[0,t] + model.sp_r[1,t])
        model.T_in = Constraint(model.nbed,model.t,rule=T_in_rule)

        def p_rule(model,i,n,s,t):
            z = model.z_mass[i,n,s,t]/sum(model.z_mass[i,n,s,t]/MW_comp[i] for i in NoComp)/MW_comp[i]
            return model.p[i,n,s,t] == P*z
        model.pressure = Constraint(NoComp,model.nbed,model.seg,model.t,rule=p_rule)

        '''
        def rho_g_rule(model,n,s,t):
            return model.rho_g[n,s,t]*model.Tg[n,s,t] == P*1e5/R
        model.rho_gas = Constraint(model.nbed,model.seg,model.t,rule=rho_g_rule)
        '''
        def r_rac_rule(model,n,s,t):
            k1 = 1.79e4 * exp(-87090/(R*model.Tc[n,s,t]))
            k2 = 2.57e16 * exp(-198464/(R*model.Tc[n,s,t]))
            return model.r_rac[n,s,t] == (k1*model.p[1,n,s,t]*(model.p[0,n,s,t]+1e-6)**1.5/(model.p[2,n,s,t]+1e-6) -k2*model.p[2,n,s,t]/(model.p[0,n,s,t]+1e-6)**1.5)*34/rho_c*4.75
        model.reactionrate = Constraint(model.nbed,model.seg,model.t,rule=r_rac_rule)

        def mass_balance_rule1(model,n,s,t):
            z_mass  = model.z_mass[2,n,s,t]
            if s == 1:
                z_mass_in = model.z_mass_bed_0[2,n,t]
                return model.m[n,t]*(z_mass - z_mass_in) == (1-eps)*rho_c*(model.r_rac[n,s,t]/3600)*A*l[int(n)]
            else:
                z_mass_in = model.z_mass[2,n,s-1,t]
                return model.m[n,t]*(z_mass - z_mass_in) == (1-eps)*rho_c*(model.r_rac[n,s,t]/3600)*A*l[int(n)]
        model.mass_balance1 = Constraint(model.nbed,model.seg,model.t,rule=mass_balance_rule1)
        
        def mass_balance_rule2(model,n,s,t):
            return sum(model.z_mass[i,n,s,t] for i in NoComp) == 1
        
        #model.mass_balance2 = Constraint(model.nbed,model.seg,model.t,rule=mass_balance_rule2)

        def mass_balance_rule3(model,n,s,t):
            if s == 1:
                return (model.z_mass_bed_0[1,n,t] - model.z_mass[1,n,s,t])/MW_comp[1] == (model.z_mass[2,n,s,t] - model.z_mass_bed_0[2,n,t])/MW_comp[2]/2
            else:
                return (model.z_mass[1,n,s-1,t] - model.z_mass[1,n,s,t])/MW_comp[1] == (model.z_mass[2,n,s,t] - model.z_mass[2,n,s-1,t])/MW_comp[2]/2
        model.mass_balance3 = Constraint(model.nbed,model.seg,model.t,rule=mass_balance_rule3)

        def mass_balance_rule4(model,n,s,t):
            if s == 1:
                return (model.z_mass_bed_0[0,n,t] - model.z_mass[0,n,s,t])/MW_comp[0] == (model.z_mass[2,n,s,t] - model.z_mass_bed_0[2,n,t])/MW_comp[2]/2*3
            else:
                return (model.z_mass[0,n,s-1,t] - model.z_mass[0,n,s,t])/MW_comp[0] == (model.z_mass[2,n,s,t] - model.z_mass[2,n,s-1,t])/MW_comp[2]/2*3
        model.mass_balance4 = Constraint(model.nbed,model.seg,model.t,rule=mass_balance_rule4)
    
        
        def heat_transfer_rule1(model,n,s,t):
                if s == 1:
                    Tg_prev = model.Tbed_0[n,t]
                else:
                    Tg_prev = model.Tg[n,s-1,t]
                #enth_rac = -4.184/MW_comp[2]*(-9184 - 7.2949*model.Tg[n,s,t] + 0.34996e-2*model.Tg[n,s,t]**2 + 0.03356e-5*model.Tg[n,s,t]**3 - 0.11625e-9*model.Tg[n,s,t]**4 - (6329.3 - 3.1619*(P*.98692)) + (14.3595 + 4.4552e-3*(P*0.98692))*model.Tg[n,s,t] - model.Tg[n,s,t]**2*(8.3395e-3 + 1.928e-6*P*0.98692) - 51.21 + 0.14215*P*0.98692)*1e-3
                MW = 1/sum (model.z_mass[i,n,s,t]/MW_comp[i] for i in NoComp)
                Pr_g = model.CP[n,s,t]*(mu_g_i*1e-3)/nambda_g_i+.001
                Re_g = D_p*(model.m[n,t]/A)/(1-eps)/(mu_g_i*1e-3)+1
                U_gc = nambda_g_i/(pi*D_p**2)*(2+1.1*(Re_g+1e-5)**0.6*(Pr_g+1e-5)**0.33)*1e-3
                #U_gc = 800
                return model.dTgdt[n,s,t]*P*1e5/R/model.Tg[n,s,t]*MW*model.CP[n,s,t] == -model.m[n,t]/eps/A*(model.Tg[n,s,t] - Tg_prev)/l[int(n)]*model.CP[n,s,t] + U_gc*1e3/eps*(model.Tc[n,s,t]-model.Tg[n,s,t])
        model.heat_transfer1 = Constraint(model.nbed,model.seg,model.t,rule=heat_transfer_rule1)

        def heat_transfer_rule2(model,n,s,t):

                enth_rac = -4.184/MW_comp[2]*(-9184 - 7.2949*model.Tg[n,s,t] + 0.34996e-2*model.Tg[n,s,t]**2 + 0.03356e-5*model.Tg[n,s,t]**3 - 0.11625e-9*model.Tg[n,s,t]**4 - (6329.3 - 3.1619*(P*.98692)) + (14.3595 + 4.4552e-3*(P*0.98692))*model.Tg[n,s,t] - model.Tg[n,s,t]**2*(8.3395e-3 + 1.928e-6*P*0.98692) - 51.21 + 0.14215*P*0.98692)*1e-3
                #enth_rac = 2900
                MW = 1/sum (model.z_mass[i,n,s,t]/MW_comp[i] for i in NoComp)
                Pr_g = model.CP[n,s,t]*(mu_g_i*1e-3)/nambda_g_i+.001
                Re_g = D_p*(model.m[n,t]/A)/(1-eps)/(mu_g_i*1e-3)+1
                U_gc = nambda_g_i/(pi*D_p**2)*(2+1.1*(Re_g+1e-5)**0.6*(Pr_g+1e-5)**0.33)*1e-3
                #U_gc = 800          
                return model.dTcdt[n,s,t] == enth_rac*1e3*(model.r_rac[n,s,t]/3600)/cp_c - U_gc*1e3/(1-eps)/rho_c/cp_c*(model.Tc[n,s,t]-model.Tg[n,s,t]) 
        model.heat_transfer2 = Constraint(model.nbed,model.seg,model.t,rule=heat_transfer_rule2)

        '''
        def separation_rule(model,t):
            K1 = 1/P_sep*((26000-8900)/50*(298.15-T_sep)+8900)
            K2 = 1/P_sep*((48000-15200)/50*(298.15-T_sep)+15200)
            K3 = 1/P_sep*10**(4.4854-926.132/(T_sep-32.98))
            return 0 == model.z[1,2,10,t]*(K1-1)/(1+model.f[t]*(K1-1)) + model.z[0,2,10,t]*(K2-1)/(1+model.f[t]*(K2-1)) + model.z[2,2,10,t]*(K3-1)/(1+model.f[t]*(K3-1))
        
        if _enable_recycle == True:   
            model.separation = Constraint(model.t,rule=separation_rule)
        '''
        def dN_sep_rule(model,i,t):
                
                return model.dN[i,t] == (model.m[2,t]*model.z_mass[i,2,10,t])/MW_comp[i] - model.Fcyc[t]*model.z_cyc[i,t]*(1+model.frac_purge[t]) - model.Fp[t]*model.z_prod[i,t]

        def z_sep_rule1(model,i,t):
            if i < 2:
                return model.z_cyc[i,t]*model.P_sep[t]/1.013 == H[i]*model.z_prod[i,t]
            else:
                return model.z_cyc[i,t]*model.P_sep[t] == model.z_prod[i,t]*10**(4.8688-1113.928/(T_sep-10.409))
            
        def z_sep_rule2(model,t):
            return sum(model.z_cyc[i,t] for i in NoComp) == 1
        
        def z_sep_rule3(model,t):
            return sum(model.z_prod[i,t] for i in NoComp) == 1
        
        def Nv_rule(model,i,t):
            
            return model.Nv_sep[t]*model.z_cyc[i,t] + model.Nl_sep[t]*model.z_prod[i,t] == model.N_sep[i,t]
            
        def P_sep_rule(model,t):

            return model.P_sep[t] == model.Nv_sep[t]/1e5/(V_sep-model.Nl_sep[t]/40080)*8.314*T_sep
        
        def P_limit_rule1(model,t):
            if t == 0 :
                return Constraint.Skip
            return model.P_sep[t] <= 200
        
        def P_limit_rule2(model,t):
            if t == 0:
                return Constraint.Skip
            return model.P_sep[t] >= 180
        
        def Nl_limit_rule1(model,t):
            if t == 0:
                return Constraint.Skip
            return model.Nl_sep[t] <= 0.6*V_sep*40080
        
        def Nl_limit_rule2(model,t):
            if t == 0:
                return Constraint.Skip
            return model.Nl_sep[t] >= 0.25*V_sep*40080

        def m_cyc_rule(model,t):
            return sum(model.Fcyc[t] * model.z_cyc[i,t] * MW_comp[i] for i in NoComp) == model.mcyc[t]
        
        def PID_Nl_rule(model,t):
            K = 0.1
            return model.Fp[t] == model.Fout_ref[t] + K*(model.Nl_sep[t] - 0.3*V_sep*40080)
        def P_diff_rule(model,t):
            return model.P_diff[t] == model.P_sep[t] - 196
        
        def PID_Nv_rule(model,t):
            KI = 0.1
            KP = 0.5
            return model.Fcyc[t] == 3*model.mt[t]/molar_weight+ KP*(model.P_sep[t]-196) # + KI*model.P_int[t])
        
        if _enable_recycle == True:
            model.m_cyc = Constraint(model.t,rule=m_cyc_rule)
            model.dN_sep = Constraint(NoComp, model.t,rule=dN_sep_rule)
            model.z_sep1 = Constraint(NoComp, model.t,rule=z_sep_rule1)
            model.z_sep2 = Constraint(model.t,rule=z_sep_rule2)
            model.z_sep3 = Constraint(model.t,rule=z_sep_rule3)
            model.Nl_limit1 = Constraint(model.t,rule=Nl_limit_rule1)
            model.Nl_limit2 = Constraint(model.t,rule=Nl_limit_rule2)
            #model.PID_Nl = Constraint(model.t,rule=PID_Nl_rule)
            #model.Int_P = Constraint(model.t,rule=P_diff_rule)
            #model.PID_Nv = Constraint(model.t,rule=PID_Nv_rule)
            model.Nv = Constraint(NoComp, model.t,rule=Nv_rule)
            model.P_sep_constr = Constraint(model.t,rule=P_sep_rule)
            model.P_limit1 = Constraint(model.t,rule=P_limit_rule1)
            model.P_limit2 = Constraint(model.t,rule=P_limit_rule2)
            
    

        #discretizing
        #discretizer = TransformationFactory('dae.collocation')
        #discretizer.apply_to(model,nfe=int(time/interval),ncp=2,scheme='LAGRANGE-RADAU')
        discretizer = TransformationFactory('dae.finite_difference') 
        discretizer.apply_to(model,nfe=int(time/interval),wrt=model.t,scheme='BACKWARD')
        #model = discretizer.reduce_collocation_points(model, var=model.sp_r, ncp=1, contset=model.t)
        #model = discretizer.reduce_collocation_points(model, var=model.Fcyc, ncp=1, contset=model.t)
        #model = discretizer.reduce_collocation_points(model, var=model.mcyc, ncp=1, contset=model.t)
        #model = discretizer.reduce_collocation_points(model, var=model.mt, ncp=1, contset=model.t)
        #model = discretizer.reduce_collocation_points(model, var=model.Fp, ncp=1, contset=model.t)
        #model = discretizer.reduce_collocation_points(model, var=model.frac_purge, ncp=1, contset=model.t)

        for t in model.t:
            index = int(t/time_action)
            model.Fin_ref[t].fix(_feed_profile[index]*molar_weight)
            #model.mt[t].value = _feed_profile[index]*molar_weight
            model.mt[t].fix(_feed_profile[index]*molar_weight)
            model.Fout_ref[t].fix(_product_profile[index])
            #model.Fout_ref[t].fix(_product_profile[index])
            model.Fp[t].value =  _product_profile[index]
            #model.P_ref[t].fix(_P_profile[index])
            #model.Nl_ref[t].fix(_Nl_profile[index])

        #initial guess
        if dn_model:
            #model.P_int[0].fix(dn_model.P_int[model.t.last()])
            for t in model.t:
                #model.Tin[t].value = value(dn_model.Tin[t])
                model.Nl_sep[t].value = value(dn_model.Nl_sep[t])
                model.Nv_sep[t].value = value(dn_model.Nv_sep[t])
                if _enable_recycle:
                    model.Fcyc[t].set_value(min(value(dn_model.Fcyc[t]),RR*600))
                    #model.Fcyc[t].fix(value(dn_model.Fcyc[t]))
                    model.mcyc[t].set_value(value(dn_model.mcyc[t]))
                    model.frac_purge[t].set_value(value(dn_model.frac_purge[t]), skip_validation = True)
                    if _init_action:
                        model.Fcyc[0].fix(min(_init_action[2],RR*600))
                        model.frac_purge[0].fix(max(0,min(_init_action[4],0.01)))
                        #model.mt[0].fix(_init_action[1]*molar_weight)
                        model.Fp[0].fix(_init_action[3])
                model.T_1[t].value = value(dn_model.T_1[t])
                for i in NoComp:
                    if _enable_recycle:
                        model.z_cyc[i,t].set_value(value(dn_model.z_cyc[i,t]), skip_validation = True)
                        model.N_sep[i,t].value = value(dn_model.N_sep[i,t])
                        model.z_prod[i,t].value = value(dn_model.z_prod[i,t])
                for n in model.valves:
                        model.sp_r[n,t].set_value(min(max(value(dn_model.sp_r[n,t]),0),1))
                        if _init_action:
                            model.sp_r[n,0].fix(min(max(_init_action[0][n],0),1))
                            
                for n in model.nbed:
                    model.Tbed_0[n,t].value = value(dn_model.Tbed_0[n,t])
                    model.m[n,t].value = value(dn_model.m[n,t])
                    for i in NoComp:
                        model.z_mass_bed_0[i,n,t].set_value(value(dn_model.z_mass_bed_0[i,n,t]), skip_validation = True)
                    for s in model.seg:
                        model.Tg[n,s,t].value = value(dn_model.Tg[n,s,t])
                        model.Tc[n,s,t].value = value(dn_model.Tc[n,s,t])
                        model.CP[n,s,t].value = value(dn_model.CP[n,s,t])
                        model.r_rac[n,s,t].value = value(dn_model.r_rac[n,s,t])
                        #model.rho_g[n,s,t].value = value(dn_model.rho_g[n,s,t])
                        for i in NoComp:
                            model.z_mass_in[i,t].set_value(value(dn_model.z_mass_in[i,t]), skip_validation = True)
                            model.z_mass[i,n,s,t].set_value(value(dn_model.z_mass[i,n,s,t]), skip_validation = True)
                            model.p[i,n,s,t].value = value(dn_model.p[i,n,s,t])
        elif st_model:
            for t in model.t:
                #model.Tin[t].value = value(st_model.Tin)
                if _enable_recycle:
                    model.Fcyc[t].set_value(min(value(st_model.Fcyc),RR*600))
                    #model.Fcyc[t].fix(value(st_model.Fcyc))
                    model.mcyc[t].set_value(value(st_model.mcyc))
                    model.N_sep[0,t].set_value(P_sep*1e5*V_sep*(1-liquid_frac)/8.314/T_sep*(1-0.078)*0.75)
                    model.N_sep[1,t].set_value(P_sep*1e5*V_sep*(1-liquid_frac)/8.314/T_sep*(1-0.078)*0.25)
                    model.N_sep[2,t].set_value(P_sep*1e5*V_sep*(1-liquid_frac)/8.314/T_sep*0.078+V_sep*liquid_frac*40080)
                    if _init_action:
                        model.frac_purge[0].fix(max(0,min(_init_action[4],0.01)))
                        model.Fcyc[0].fix(min(_init_action[2],RR*600))
                        #model.mt[0].fix(_init_action[1]*molar_weight)
                        model.Fp[0].fix(_init_action[3])
                model.T_1[t].value = value(st_model.T_1)
                for i in NoComp:
                    if _enable_recycle:
                        model.z_cyc[i,t].value = value(st_model.z_cyc[i])
                        model.z_prod[i,t].value = value(st_model.z_prod[i])
                for n in model.valves:
                    model.sp_r[n,t].set_value(min(max(value(st_model.sp_r[n]),0),1))
                    if _init_action:
                        model.sp_r[n,0].fix(min(max(_init_action[0][n],0),1))

                for n in model.nbed:
                    model.Tbed_0[n,t].value = value(st_model.Tbed_0[n])
                    model.m[n,t].value = value(st_model.m[n])
                    for i in NoComp:
                        model.z_mass_bed_0[i,n,t].value = value(st_model.z_mass_bed_0[i,n])
                    for s in model.seg:
                        if t > 0:
                            model.Tg[n,s,t].value = value(st_model.Tg[n,s])
                            model.Tc[n,s,t].value = value(st_model.Tc[n,s])
                        model.CP[n,s,t].value = value(st_model.CP[n,s])
                        model.r_rac[n,s,t].value = value(st_model.r_rac[n,s])
                        #model.rho_g[n,s,t].value = value(st_model.rho_g[n,s])
                        for i in NoComp:
                            model.z_mass_in[i,t].value = value(st_model.z_mass_in[i])
                            model.z_mass[i,n,s,t].value = value(st_model.z_mass[i,n,s])
                            model.p[i,n,s,t].value = value(st_model.p[i,n,s])

        def _init(model):
            for n in model.nbed:
                for s in model.seg:
                    yield model.Tg[n,s,0] == Tg0[n,s-1]
                    yield model.Tc[n,s,0] == Tc0[n,s-1]
            yield model.P_sep[0] == P0
            yield model.Nl_sep[0] == Nl0
            for t in model.t:
                yield model.sp_r[0,t] >= 0.05
                if t != 0:
                    yield model.Fp[t] >= 0.9*model.Fp[model.t.prev(t)]
                    yield model.Fp[t] <= 1.1*model.Fp[model.t.prev(t)]
                    yield model.Fcyc[t] >= 0.9*model.Fcyc[model.t.prev(t)]
                    yield model.Fcyc[t] <= 1.1*model.Fcyc[model.t.prev(t)]
                    #yield model.P_sep[t] - model.P_sep[model.t.prev(t)] <= 1
                    #yield model.P_sep[t] - model.P_sep[model.t.prev(t)] >= -1
                    for n in model.nbed:
                        yield model.sp_r[n,t] - model.sp_r[n,model.t.prev(t)] <= 0.1
                        yield model.sp_r[n,t] - model.sp_r[n,model.t.prev(t)] >= -0.1
                        for s in model.seg:
                            yield model.Tc[n,s,t] - model.Tc[n,s,model.t.prev(t)] <= 5
                            yield model.Tc[n,s,t] - model.Tc[n,s,model.t.prev(t)] >= -5
                            yield model.Tg[n,s,t] - model.Tg[n,s,model.t.prev(t)] <= 5
                            yield model.Tg[n,s,t] - model.Tg[n,s,model.t.prev(t)] >= -5
            yield model.z_prod[2,0] == z_NH3_prod
            yield model.Nl_sep[model.t.last()] == liquid_frac*V_sep*40080
            #for n in model.nbed:
            #    yield model.Tbed_0[n,model.t.last()] == value(st_model.Tbed_0[n])
            #yield model.Fp[model.t.last()] == value(st_model.Fp)
            #yield model.P_sep[model.t.last()] == 196
        #            yield model.F[n,s,0] == F0[n,s-1]
        #            if z0[0,0,0] != 0:
        #                for comp in range(3):
        #                    yield model.z[comp,n,s,0] == z0[n,s-1,comp]
        model.init_conditions = ConstraintList(rule=_init)

        def _intX_Fp(model,t):
            #return (model.Nl_sep[t]/40080 - liquid_frac*V_sep)**2 + (model.Fcyc[t]*model.frac_purge[t])**2*0.01 + sum((model.Tbed_0[n,t] - value(st_model.Tbed_0[n]))**2 for n in model.nbed) + (model.P_sep[t]-196)**2
            if t == 0:
                return sum((model.Tbed_0[n,t] - value(st_model.Tbed_0[n]))**2*0.6 + (model.Tg[n,10,t] - value(st_model.Tg[n,10]))**2*0.25 for n in model.nbed) + (model.P_sep[t]-196)**2 + (model.Fcyc[t]*model.frac_purge[t])**2*0.001 #+ (model.Nl_sep[t]/40080 - 0.3*V_sep)**2*10
            else:
                return sum((model.Tbed_0[n,t] - value(st_model.Tbed_0[n]))**2*0.6 + (model.Tg[n,10,t] - value(st_model.Tg[n,10]))**2*0.25 for n in model.nbed) + (model.Fp[t] - model.Fp[model.t.prev(t)])**2*0.01 + (model.Fcyc[t] - model.Fcyc[model.t.prev(t)])**2*5e-3 + (model.P_sep[t]-196)**2 + (model.Fcyc[t]*model.frac_purge[t])**2*0.001 # + (model.Nl_sep[t]/40080 - 0.3*V_sep)**2*10
        model.intX_Fp = Integral(model.t,wrt=model.t, rule=_intX_Fp)

        def _obj(model):

            return model.intX_Fp 
        
        model.obj = Objective(
                rule=_obj,
                sense = minimize)

        opt = SolverFactory('ipopt')
        opt.options['max_iter']= 10000
        opt.options['linear_solver'] = 'ma86'
        opt.options['hsllib'] = 'C:/Users/kong0225/Anaconda3/Library/bin/libhsl.dll'
        opt.options['warm_start_init_point'] = 'yes'
        #opt.options['dependency_detector'] = 'ma28'
        opt.options['dependency_detection_with_rhs'] = 'yes'
        opt.options['jac_d_constant'] = 'yes'
        opt.options['resto_failure_feasibility_threshold'] = 0
        #opt.options['line_search_method'] = 'cg-penalty'
        #opt.options['neg_curv_test_reg'] = 'yes'
        #opt.options['perturb_always_cd'] = 'yes'
        #opt.options['warm_start_entire_iterate'] = 'yes'
        #opt.options['max_cpu_time']= time_implementation
        #opt.options['tol']= 1e-5
        #opt.options['constr_viol_tol'] = 1e-6
        #opt.options['acceptable_tol']= 1e-3
        #opt.options['CompIIS'] = 1
        #opt.options['max_cpu_time'] = 20
        #opt.options['halt_on_ampl_error']='yes'
        #model.pprint()

        result = opt.solve(model,tee=True,keepfiles=False)

        return {'model':model,'status': result.solver.status,'termination_condition':result.solver.termination_condition, 'Time':result['Solver'][0]['Time']}

    #observations, Reward = env.reset()
    terminated = False
    '''
    Tg0_1 = 700*np.ones(10)
    Tg0_2 = 700*np.ones(10)
    Tg0_3 = 700*np.ones(10)
    Tc0_1 = 700*np.ones(10)
    Tc0_2 = 700*np.ones(10)
    Tc0_3 = 700*np.ones(10)
    Tg = np.stack((Tg0_1,Tg0_2,Tg0_3))
    Tc = np.stack((Tc0_1,Tc0_2,Tc0_3))
    '''
    acc_production = 0
    time_step = 0
    Fp_schedule = product_profile[time_step]
    
    st = Optsteady(L=L, D=D, 
                   Fp_schedule = Fp_schedule, 
                   Ft=Ft, Tt=Tt, HX_U=HX_U, HX_A=HX_A,
                    T_init=np.array([700,700,700]), 
                    z_init=np.array([0.6,0.2,0.2]),
                    tol=1e-5,
                    RR=RR,P=P,P_sep=P_sep,T_sep=T_sep, _enable_recycle = enable_recycle) #steady state
    if (st['status'] == SolverStatus.ok) and (st['termination_condition'] == TerminationCondition.optimal):
        st_model = st['model']
        st_Tg = st_model.Tg.extract_values()
        st_Tc = st_model.Tc.extract_values()
        st_z_prod = st_model.z_prod.extract_values()
        st_model.Tbed_0.pprint()
        st_model.Fp.pprint()
    record_hist = []
    action_buffer = []
    init_action = None
    next_action = None
    dn_model = None
    P0 = P_sep
    Nl0 = V_sep*liquid_frac*40080
    z_NH3_prod = st_z_prod[2]
    Tg = np.zeros([3,10])
    Tc = np.zeros([3,10])
    for i_bed in range(3):
        for i_seg in range(10):
                Tg[i_bed,i_seg] = st_Tg[(i_bed,i_seg+1)]
                Tc[i_bed,i_seg] = st_Tc[(i_bed,i_seg+1)]
    env = NH3reactor(length=L, diameter=D, F_in=feed_profile, T_in=Tt, HX_A=HX_A,HX_U=HX_U,max_RR=RR,
                            interval=time_action, dt=1,Time=total_time_horizon,
                            init_Tg = Tg, init_Tc = Tc, z_prod_init=z_NH3_prod,
                            num_segs=10, _enable_recycle = enable_recycle,
                            T_sep=T_sep, P_sep=P_sep, V_sep=V_sep, liquid_frac_sep=liquid_frac,
                            noise_Fin=0.0, noise_Tin=0.0, noise_Tsep = 0.0, noise_type='normal',
                            #random_seed=np.random.randint(1000)
                            random_seed=1000
                            )

    def get_action(indice, action_buffer):
        for i in indice:
            action_sp_r = np.zeros(4)
            for j in range(4):
                action_sp_r[j] = sp_r[(j,i*time_action)]
            action_Fcyc = Fcyc[i*time_action]
            action_Ft = mt[i*time_action]/molar_weight
            action_Fp = Fp[i*time_action]
            action_frac_purge = frac_purge[i*time_action]
            if enable_recycle:
                action = [action_sp_r, action_Ft, action_Fcyc, action_Fp, action_frac_purge]
            else:
                action = [action_sp_r, action_Ft]
            action_buffer.append(action)
            #print(action_buffer)
        return action_buffer

    while not terminated:
        time_feasibility = -1000
        time_MPC = 9999
        # if schedule changes, obtain a new steady state 
        if Ft != feed_profile[time_step]:
            Ft = feed_profile[time_step]
            Fp_schedule = product_profile[time_step]
            st = Optsteady(L=L, D=D, 
                           Fp_schedule = Fp_schedule, 
                           Ft=Ft, Tt=Tt, HX_A=HX_A,HX_U=HX_U,
                    T_init=np.array([700,700,700]), 
                    z_init=np.array([0.6,0.2,0.2]),
                    tol=1e-5,
                    RR=RR,P=P,P_sep=P_sep,T_sep=T_sep, _enable_recycle = enable_recycle) #steady state
            if (st['status'] == SolverStatus.ok) and (st['termination_condition'] == TerminationCondition.optimal):
                st_model = st['model']
            #dn_model = None
        
        '''
        # ML prediction
        observation = {
                "feed_schedule": feed_profile[time_step: time_step+num_steps_prediction+1], 
                "product_schedule": product_profile[time_step: time_step+num_steps_prediction+1],
                "obs_Tg": Tg.copy(), #gas temperatures, size = 3*10
                "obs_Tc": Tc.copy(), #catalyst temperatures, size = 3*10
                "obs_feed_flowrate": observations["feed_flowrates"], #feed flowrates at each reactor bed, size = 3
                "obs_out_flowrate": observations["flowrates"], #output flowrates at each reactor bed, size = 3
                "obs_Fp": Reward/time_action, #production rate
        }
        #print(observation)
        y_pred = ML_predict([observation])
        _predictive_action = {'spr_1': [y_pred[0], y_pred[0], y_pred[1], y_pred[2]],
                            'spr_2': [max(0,y_pred[3]), max(0,y_pred[3]), max(0,y_pred[4]),max(0,y_pred[5])],
                            'spr_3': [y_pred[6],y_pred[6], y_pred[7],y_pred[8]],
                            'spr_4': [y_pred[9],y_pred[9], y_pred[10],  y_pred[11]],
                            'Ft': [y_pred[12],y_pred[12], y_pred[13],  y_pred[14]],
                            'Fcyc': [y_pred[15],y_pred[15], y_pred[16],  y_pred[17]]}
        #print(_predictive_action)

        # solve feasibility problem
        fm = prediction_feasibility(L,D,Tg,Tc,P,P_sep,T_sep,zt,time_action,_predictive_action,
                                    Tt = 313,
                                   HX_U = 500,
                                    HX_A = 100,
                                    RR = 3, #recycle ratio
                                    dn_model = dn_model, st_model = st['model'], _enable_recycle = True,
                                    time = time_prediction,interval=time_action,
                                    _init_action = init_action)
        if fm['termination_condition'] == TerminationCondition.optimal:
            time_feasibility = fm['Time']
            print('Feasible solution found. Time consumption: ', fm['Time'])
            dn_model = fm['model']
            if time_feasibility > 100:
                with open('bad_feasibility_'+str(time_step)+'.pkl', mode='wb') as file:
                    cloudpickle.dump(_predictive_action, file)
        else:
            with open('bad_feasibility_'+str(time_step)+'.pkl', mode='wb') as file:
                    cloudpickle.dump(_predictive_action, file)
        ''' 
        
        MPC_predict = prediction(Tg, Tc, P0, Nl0, z_NH3_prod,
                                    _feed_profile = feed_profile[time_step : time_step+num_steps_prediction+1],
                                    _product_profile = product_profile[time_step : time_step+num_steps_prediction+1],
                                    _P_profile = P_profile[time_step : time_step+num_steps_prediction+1],
                                    _Nl_profile = Nl_profile[time_step : time_step+num_steps_prediction+1],
                                    st_model = st_model, 
                                    dn_model = None,
                                    _enable_recycle = enable_recycle,
                                    time = time_prediction, interval=time_action,
                                    _init_action = init_action, _next_action = None)
        if MPC_predict['termination_condition'] == TerminationCondition.optimal:
            #print('MPC solved at step ', time_step, ' Time consumption: ', MPC_predict['Time'])
            time_MPC = MPC_predict['Time']

        dn_model = MPC_predict['model']
        sp_r = dn_model.sp_r.extract_values()
        mt = dn_model.mt.extract_values()
        Fcyc = dn_model.Fcyc.extract_values()
        Fp = dn_model.Fp.extract_values()
        frac_purge = dn_model.frac_purge.extract_values()
        Tbed0_sp = st_model.Tbed_0.extract_values()
        Tg_sp = st_model.Tg.extract_values()
        full_action = get_action(indice = list(range(1,1+num_steps_prediction)), action_buffer=[])
        if action_buffer:
            action_buffer = get_action(indice = list(range(2,2+num_steps_implementation)), action_buffer=action_buffer)
        else:
            action_buffer = get_action(indice = list(range(1,2+num_steps_implementation)), action_buffer=action_buffer)
        
        for i in range(num_steps_implementation):
            print(full_action[i])
            Tg, Tc, P0, Nl0, obs_feed_flowrate, obs_out_flowrate, obs_concentration, z_prod, Reward, terminated = env.step(full_action[i], MPC_purpose=True)
            z_NH3_prod = z_prod[2]
            print('pressure ', P0, 'liquid volume', Nl0/40080)
            acc_production += Reward
            init_action = full_action[i]
            #init_action = action_buffer.pop(0)
            #next_action = action_buffer[0]
            time_step += 1

            record = {
                "feed_schedule": feed_profile[time_step-1 : time_step+num_steps_prediction], 
                "product_schedule": product_profile[time_step-1 : time_step+num_steps_prediction],
                "Tbed0_sp": Tbed0_sp,
                "Tg_sp": Tg_sp,
                "obs_Tg": Tg.copy(), #gas temperatures, size = 3*10
                "obs_Tc": Tc.copy(), #catalyst temperatures, size = 3*10
                "obs_feed_flowrate": obs_feed_flowrate, #feed flowrates at each reactor bed, size = 3
                "obs_out_flowrate": obs_out_flowrate, #output flowrates at each reactor bed, size = 3
                "obs_concentration": obs_concentration,
                "obs_Fp": Reward/time_action, #production rate
                "obs_P": P0, #pressure in the separator
                "obs_L": Nl0/40080, #liquid volume in the separator
                "obs_z_prod": z_prod,
                "spr_1": [full_action[0][0][0], full_action[1][0][0],full_action[2][0][0]],#action 1
                "spr_2": [full_action[0][0][1], full_action[1][0][1],full_action[2][0][1]], #action 2
                "spr_3": [full_action[0][0][2], full_action[1][0][2],full_action[2][0][2]], #action 3
                "spr_4": [full_action[0][0][3], full_action[1][0][3],full_action[2][0][3]], #action 4
                "Ft": [full_action[0][1], full_action[1][1],full_action[2][1]], #action 5
                "Fcyc": [full_action[0][2], full_action[1][2],full_action[2][2]], #action 6
                "Fp": [full_action[0][3], full_action[1][3],full_action[2][3]],
                "purge_ratio": [full_action[0][4], full_action[1][4],full_action[2][4]],
                "time_feasibility": time_feasibility,
                "time_MPC" : time_MPC
            }
            if True: #MPC_predict['termination_condition'] == TerminationCondition.optimal:
                if not data_set:
                    try:
                        data_set = pickle.load(open(file_name,'rb'))
                    except (EOFError, FileNotFoundError):
                        data_set = []
                data_set.append(record)
                pickle.dump(data_set, open(file_name,'wb'))
                #print('from process ', index, 'dataset_size ', len(data_set))

    return len(data_set)

if __name__ == '__main__':
    MPC_running(0000, _seed = 123)
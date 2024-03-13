# This is a steady-state optimization of ammonia synthesis reactors
# Author info -- Baiwen Kong, Octobor 2022, University of Minnesota
# Email -- kong0225@umn.edu

from math import pi
from pyomo.environ import *
from pyomo.dae import *
import numpy as np
# from cyipopt import *


NoComp  = range(3) #[H2 N2 NH3]
MW_comp = np.array([2, 28.01, 17.03])*1e-3 #molecular weight
R       = 8.314
D_p     = .00285 #catalyst particle diameter
cp_c    = 1100 #catalyst heat capacity J/kg/K
rho_c   = 2200 #catalyst denstiy 2200 kg/m3
eps      = .33 #bed void fraction
mu_g_i  = 0.028 #cP viscosity
nambda_g_i = 0.231 #W/mK thermal conductivity

# calculate heat capacities
'''
C1 = np.array([33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974]) #H2
C2 = np.array([19.50583, 19.88705, -8.598535, 1.369784, .527601, -4.935202, 212.39])#N2
C3 = np.array([19.99563, 49.77119, -15.37599, 1.921168, .189174, -53.30667, -45.89806])#NH3
CP_comp = np.zeros(3)
C = np.array([C1,C2,C3])

def Cp(T,P):
    C1 = np.array([33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974]) #H2
    C2 = np.array([19.50583, 19.88705, -8.598535, 1.369784, .527601, -4.935202, 212.39])#N2
    C3 = np.array([19.99563, 49.77119, -15.37599, 1.921168, .189174, -53.30667, -45.89806])#NH3
    CP_comp = np.zeros(3)
    C = np.array([C1,C2,C3])
    for i in range(2):
         CP_comp[i] = (C[i,0] + C[i,1]*(T/1000) + C[i,2]*(T/1000)**2 + C[i,3]*(T/1000)**3 + C[i,4]/(T/1000)**2)
    CP_comp[2] = 4.184*(6.5846 - 0.61251e-2*T + 0.23663e-5*T**2 - 1.5981e-9*T**3 + (96.1678-0.067571*P*0.98692) + (-.2225 + 1.6847e-4*P*0.98692)*T + (1.289e-4 - 1.0095e-7*P*.98692)*T**2)
    return CP_comp
'''
# design conditions
#Ft = 4 #input flowrate of H2, mol/s
#zt = np.array([0.75,0.25,0]) #input molar fractions

#initial conditions
#T_init = np.array([580,680,700]) #for reactor bed 1,2, and 3
#z_init = np.array([0.525,0.175,0.3]) #for H2, N2, and NH3

def Optsteady(L, D, Ft,  Tt, T_init, z_init, Fp_schedule = None,HX_U=600, HX_A=100, P=200,T_sep=313.34,T_1=600,P_sep = 200,n_bed=3,n_seg=11,RR=3,zt=np.array([0.75,0.25,1e-6]),tol=1e-5, _enable_recycle = True):
    l = L/(n_seg-1)
    A = D**2/4*pi
    
    CP_comp = np.array([14.6,1.1,3.16])*1000
    K1 = 1/P_sep*((26000-8900)/50*(298.15-T_sep)+8900)
    K2 = 1/P_sep*((48000-15200)/50*(298.15-T_sep)+15200)
    K3 = 1/P_sep*10**(4.4854-926.132/(T_sep-32.98))
    Henry_const = np.array([[-3.68607, -2.29337],
                        [0.596736*1e4, 0.5294740*1e4],
                        [-0.642828*1e6,-0.521881*1e6]])
    H = np.exp(np.matmul(Henry_const.transpose(),np.array([1,1/T_sep,1/T_sep**2])))
    mt = Ft*sum(MW_comp*zt)
    z_mass_t = Ft*zt*MW_comp/mt
    CPt = np.dot(CP_comp,z_mass_t)

    # model formulation
    model = ConcreteModel()
    model.nbed = RangeSet(0,n_bed-1)
    model.valves = RangeSet(0,n_bed)
    model.seg = RangeSet(1,n_seg-1)
    model.sp_r = Var(model.valves,bounds=(0,1),initialize=0.33)
    model.Tin = Var(bounds=(500,800),initialize=550)
    model.Tbed_0 = Var(model.nbed, bounds = (500,800),initialize=550)
    model.z_mass_bed_0 = Var(NoComp, model.nbed, bounds=(0,1),initialize=0.33)
    model.z_mass_in = Var(NoComp,bounds=(0,1),initialize=0.33)
    model.CPin = Var(bounds=(0,100000),initialize=28)
    model.Tg = Var(model.nbed,model.seg,bounds=(500,800))
    model.Tc = Var(model.nbed,model.seg,bounds=(500,800))
    #model.z = Var(NoComp,model.nbed,model.seg,bounds=(0,1),initialize=0.33)
    model.z_mass = Var(NoComp,model.nbed,model.seg,bounds=(0,1),initialize=0.33)
    model.P = Var(model.nbed,model.seg,bounds=(100,200), initialize = P)
    model.p = Var(NoComp,model.nbed,model.seg,bounds=(0,200),initialize = P*0.33)
    model.CP = Var(model.nbed,model.seg,bounds=(0,20000),initialize=2800)
    model.r_rac = Var(model.nbed,model.seg,bounds=(-10000,10000),initialize=30)
    model.m = Var(model.nbed,bounds=(0,300),initialize=80)
    model.rho_g = Var(model.nbed,model.seg,bounds=(0,10000),initialize=2000)
    if _enable_recycle == True:
        model.mcyc = Var(bounds=(0,300),initialize=60)
        model.Fcyc = Var(bounds=(600,RR*600),initialize=RR*Ft)
        model.frac_purge = Var(bounds=(0,0.1), initialize = 0.01)
        model.z_cyc = Var(NoComp,bounds=(0,1),initialize=0.33)
        model.z_mass_cyc = Var(NoComp,bounds=(0,1),initialize=0.33)
        model.z_prod = Var(NoComp,bounds=(0,1),initialize=0.33)
        model.CPcyc = Var(bounds=(0,None),initialize=2800)
    else:
        model.mcyc = Param(initialize=0)
        model.zcyc = Param(NoComp,initialize=1/3)
        model.CPcyc = Param(initialize=28)
    model.Fp = Var(initialize=Ft/2,bounds=(0,Ft))
    #model.T_1 = Param(initialize=T_1) #inlet temperature to the first bed
    model.T_1 = Var(initialize=T_1,bounds=(500,800)) #inlet temperature to the first bed
    model.tol = Param(initialize=tol) #tolarence of dT/dt

    #initialization
    for n in model.nbed:
        for s in model.seg:
            model.Tc[n,s].value = T_init[n]
            model.Tg[n,s].value = T_init[n]
            for i in NoComp:
                model.z_mass[i,n,s].value = z_init[i]
                model.p[i,n,s].value = z_init[i]*P
    #constraints
    def HeatCP_rule(model,n,s):
        
        return model.CP[n,s] == sum(CP_comp[i]*model.z_mass[i,n,s] for i in NoComp)
    
    model.HeatCP = Constraint(model.nbed,model.seg,rule=HeatCP_rule)
    
    def HeatCPin_rule(model):

        return model.CPin == sum(CP_comp[i]*model.z_mass_in[i] for i in NoComp)
    model.HeatCPin = Constraint(rule=HeatCPin_rule)

    def HeatCPcyc_rule(model):
        
        return model.CPcyc == sum(CP_comp[i]*model.z_mass_cyc[i] for i in NoComp)
    if _enable_recycle == True:
        model.HeatCPcyc = Constraint(rule=HeatCPcyc_rule)
    

    def F_total_rule(model):
        return sum(model.sp_r[n] for n in model.valves) == 1
    model.F_total = Constraint(rule=F_total_rule)

    def T_total_rule(model):
        return model.Tin == Tt
    model.T_total = Constraint(rule=T_total_rule)

    def z_total_rule(model,i):
        return model.z_mass_in[i]*(model.mcyc+ mt) == (model.z_cyc[i]*model.Fcyc + Ft*zt[i])*MW_comp[i]
    model.z_total = Constraint(NoComp,rule=z_total_rule)
    
    def F_in_rule(model,n):
        if n > 0:
            return model.m[n] == model.sp_r[n+1]*(mt + model.mcyc) + model.m[n-1]
        else:
            return model.m[n] == (model.sp_r[0]+model.sp_r[1])*(mt + model.mcyc)
    model.F_in = Constraint(model.nbed,rule=F_in_rule)

    def z_in_rule(model,i,n):
        if n > 0:
            return model.z_mass_bed_0[i,n]*model.m[n] == model.z_mass_in[i]*model.sp_r[n+1]*(mt + model.mcyc) + model.z_mass[i,n-1,n_seg-1]*model.m[n-1]
        else:
            return model.z_mass_bed_0[i,n] == model.z_mass_in[i]
    model.z_in = Constraint(NoComp,model.nbed,rule=z_in_rule)

    def T_1_rule(model):

        Fcold = model.sp_r[0]*(mt + model.mcyc)
        Fhot = model.m[2]
        Cmin = Fcold*model.CPin
        Cmax = Fhot*model.CP[2,10]
        Cr = Cmin/Cmax
        qmax = Cmin*(model.Tg[2,10]-model.Tin)
        NTU = HX_U*HX_A/Cmin
        #epsilon = (1-exp(-NTU*(1-Cr)))/(1-Cr*exp(-NTU*(1-Cr)))
        epsilon = NTU/(1+NTU)
        q = qmax * epsilon
        return model.T_1 == q/(Fcold*model.CPin) + model.Tin
    model.T_1_eq = Constraint(rule=T_1_rule)

    def delta_T_rule(model):
            return model.Tg[2,10] >= model.Tin + 20
    model.delta_T = Constraint(rule=delta_T_rule)

    def T_in_rule(model,n):
        if n > 0:
            return model.sp_r[n+1]*(mt + model.mcyc)*model.CPin*model.Tin + model.m[n-1]*model.CP[n-1,10]*model.Tg[n-1,n_seg-1] == model.m[n]*model.Tbed_0[n]*model.CP[n,1]
            #return model.sp_r[n]*(Ft + model.Fcyc)*model.Tin + model.F[n-1,n_seg-1]*model.Tg[n-1,n_seg-1] == model.Fbed_0[n]*model.Tbed_0[n]

        else:
            return model.T_1*model.sp_r[0] + model.Tin*model.sp_r[1] == model.Tbed_0[n]*(model.sp_r[0] + model.sp_r[1])
    model.T_in = Constraint(model.nbed,rule=T_in_rule)

    def T_successive_rule1(model,n,s):
        if s == 1:
            return model.Tg[n,s] - model.Tbed_0[n] <= 10
        else:
            return model.Tg[n,s] - model.Tg[n,s-1] <= 10
    
    def T_successive_rule2(model,n,s):
        if s == 1:
            return model.Tg[n,s] - model.Tbed_0[n] >= -10
        else:
            return model.Tg[n,s] - model.Tg[n,s-1] >= -10
        
    model.T_successive_1 = Constraint(model.nbed,model.seg,rule=T_successive_rule1)
    model.T_successive_2 = Constraint(model.nbed,model.seg,rule=T_successive_rule2)

    def P_drop_rule(model,n,s):
        MW = 1/sum (model.z_mass[i,n,s]/MW_comp[i] for i in NoComp)
        if s == 1:
            if n == 0:
                return model.P[n,s] == P
            return model.P[n,s] == model.P[n-1,n_seg-1]
        else:
            return model.P[n,s] == model.P[n,s-1] - model.m[n]/(model.rho_g[n,s]*MW)/A/D_p * (1-eps)/eps**3*(150*(1-eps)*(mu_g_i*1e-3)/D_p + 1.75*model.m[n]/A)*l[int(n)]/1e5
    model.P_drop = Constraint(model.nbed,model.seg,rule=P_drop_rule)

    def p_rule(model,i,n,s):
        z = model.z_mass[i,n,s]/sum(model.z_mass[i,n,s]/MW_comp[i] for i in NoComp)/MW_comp[i]
        return model.p[i,n,s] == model.P[n,s]*z
    model.pressure = Constraint(NoComp,model.nbed,model.seg,rule=p_rule)


    def rho_g_rule(model,n,s):
        return model.rho_g[n,s]*model.Tg[n,s] == P*1e5/R
    model.rho_gas = Constraint(model.nbed,model.seg,rule=rho_g_rule)

    def r_rac_rule(model,n,s):
        k1 = 1.79e4 * exp(-87090/(R*model.Tc[n,s]))
        k2 = 2.57e16 * exp(-198464/(R*model.Tc[n,s]))
        return model.r_rac[n,s] == (k1*model.p[1,n,s]*(model.p[0,n,s]+1e-5)**1.5/(model.p[2,n,s]+1e-5) -k2*model.p[2,n,s]/(model.p[0,n,s]+1e-5)**1.5)*34/rho_c*4.75
    model.reactionrate = Constraint(model.nbed,model.seg,rule=r_rac_rule)

    def mass_balance_rule1(model,n,s):

        if s == 1:
            z_mass_in = model.z_mass_bed_0[2,n]
        else:
            z_mass_in = model.z_mass[2,n,s-1]
        return model.m[n]*(model.z_mass[2,n,s] - z_mass_in) == (1-eps)*rho_c*(model.r_rac[n,s]/3600)*A*l[int(n)]
    
    model.mass_balance1 = Constraint(model.nbed,model.seg,rule=mass_balance_rule1)
    

    def mass_balance_rule3(model,n,s):
        if s == 1:
            return (model.z_mass_bed_0[1,n] - model.z_mass[1,n,s])/MW_comp[1] == (model.z_mass[2,n,s] - model.z_mass_bed_0[2,n])/MW_comp[2]/2
        else:
            return (model.z_mass[1,n,s-1] - model.z_mass[1,n,s])/MW_comp[1] == (model.z_mass[2,n,s] - model.z_mass[2,n,s-1])/MW_comp[2]/2
    model.mass_balance3 = Constraint(model.nbed,model.seg,rule=mass_balance_rule3)

    def mass_balance_rule4(model,n,s):
        if s == 1:
            return (model.z_mass_bed_0[0,n] - model.z_mass[0,n,s])/MW_comp[0] == (model.z_mass[2,n,s] - model.z_mass_bed_0[2,n])/MW_comp[2]/2*3
        else:
            return (model.z_mass[0,n,s-1] - model.z_mass[0,n,s])/MW_comp[0] == (model.z_mass[2,n,s] - model.z_mass[2,n,s-1])/MW_comp[2]/2*3
    model.mass_balance4 = Constraint(model.nbed,model.seg,rule=mass_balance_rule4)


    def heat_transfer_rule1(model,n,s):
        #enth_rac = -4.184/MW_comp[2]*(-9184 - 7.2949*model.Tg[n,s,t] + 0.34996e-2*model.Tg[n,s,t]**2 + 0.03356e-5*model.Tg[n,s,t]**3 - 0.11625e-9*model.Tg[n,s,t]**4 - (6329.3 - 3.1619*(P*.98692)) + (14.3595 + 4.4552e-3*(P*0.98692))*model.Tg[n,s,t] - model.Tg[n,s,t]**2*(8.3395e-3 + 1.928e-6*P*0.98692) - 51.21 + 0.14215*P*0.98692)*1e-3
        Pr_g = model.CP[n,s]*(mu_g_i*1e-3)/nambda_g_i+.001
        Re_g = D_p*(model.m[n]/A)/(1-eps)/(mu_g_i*1e-3)+1
        U_gc = nambda_g_i/(pi*D_p**2)*(2+1.1*(Re_g+1e-5)**0.6*(Pr_g+1e-5)**0.33)*1e-3
        #U_gc = 1000
        MW = 1/sum (model.z_mass[i,n,s]/MW_comp[i] for i in NoComp)
        if s == model.seg.first():
            return model.tol*model.rho_g[n,s]*model.CP[n,s]*MW >= -model.m[n]/eps/A*((model.Tg[n,s] - model.Tbed_0[n])/l[int(n)])*model.CP[n,s] + U_gc*1e3/eps*(model.Tc[n,s]-model.Tg[n,s])
        else:
            return model.tol*model.rho_g[n,s]*model.CP[n,s]*MW >= -model.m[n]/eps/A*((model.Tg[n,s] - model.Tg[n,s-1])/l[int(n)])*model.CP[n,s] + U_gc*1e3/eps*(model.Tc[n,s]-model.Tg[n,s])
    model.heat_transfer1 = Constraint(model.nbed,model.seg,rule=heat_transfer_rule1)

    def heat_transfer_rule2(model,n,s):
            enth_rac = -4.184/MW_comp[2]*(-9184 - 7.2949*model.Tg[n,s] + 0.34996e-2*model.Tg[n,s]**2 + 0.03356e-5*model.Tg[n,s]**3 - 0.11625e-9*model.Tg[n,s]**4 - (6329.3 - 3.1619*(P*.98692)) + (14.3595 + 4.4552e-3*(P*0.98692))*model.Tg[n,s] - model.Tg[n,s]**2*(8.3395e-3 + 1.928e-6*P*0.98692) - 51.21 + 0.14215*P*0.98692)*1e-3
            #enth_rac = 2500
            Pr_g = model.CP[n,s]*(mu_g_i*1e-3)/nambda_g_i+.001
            Re_g = D_p*(model.m[n]/A)/(1-eps)/(mu_g_i*1e-3)+1
            U_gc = nambda_g_i/(pi*D_p**2)*(2+1.1*(Re_g+1e-5)**0.6*(Pr_g+1e-5)**0.33)*1e-3
            #U_gc = 1000
            return model.tol >= enth_rac*1e3*(model.r_rac[n,s]/3600)/cp_c - U_gc*1e3/(1-eps)/rho_c/cp_c*(model.Tc[n,s]-model.Tg[n,s])
    model.heat_transfer2 = Constraint(model.nbed,model.seg,rule=heat_transfer_rule2)

    def heat_transfer_rule3(model,n,s):
        #enth_rac = -4.184/MW_comp[2]*(-9184 - 7.2949*model.Tg[n,s,t] + 0.34996e-2*model.Tg[n,s,t]**2 + 0.03356e-5*model.Tg[n,s,t]**3 - 0.11625e-9*model.Tg[n,s,t]**4 - (6329.3 - 3.1619*(P*.98692)) + (14.3595 + 4.4552e-3*(P*0.98692))*model.Tg[n,s,t] - model.Tg[n,s,t]**2*(8.3395e-3 + 1.928e-6*P*0.98692) - 51.21 + 0.14215*P*0.98692)*1e-3
        MW = 1/sum (model.z_mass[i,n,s]/MW_comp[i] for i in NoComp)
        Pr_g = model.CP[n,s]*(mu_g_i*1e-3)/nambda_g_i+.001
        Re_g = D_p*(model.m[n]/A)/(1-eps)/(mu_g_i*1e-3)+1
        U_gc = nambda_g_i/(pi*D_p**2)*(2+1.1*(Re_g+1e-5)**0.6*(Pr_g+1e-5)**0.33)*1e-3
        #U_gc = 1000
        if s == model.seg.first():
            return -model.tol*model.rho_g[n,s]*model.CP[n,s]*MW <= -model.m[n]/eps/A*((model.Tg[n,s] - model.Tbed_0[n])/l[int(n)])*model.CP[n,s] + U_gc*1e3/eps*(model.Tc[n,s]-model.Tg[n,s])
        else:
            return -model.tol*model.rho_g[n,s]*model.CP[n,s]*MW <= -model.m[n]/eps/A*((model.Tg[n,s] - model.Tg[n,s-1])/l[int(n)])*model.CP[n,s] + U_gc*1e3/eps*(model.Tc[n,s]-model.Tg[n,s])
    model.heat_transfer3 = Constraint(model.nbed,model.seg,rule=heat_transfer_rule3)

    def heat_transfer_rule4(model,n,s):

            enth_rac = -4.184/MW_comp[2]*(-9184 - 7.2949*model.Tg[n,s] + 0.34996e-2*model.Tg[n,s]**2 + 0.03356e-5*model.Tg[n,s]**3 - 0.11625e-9*model.Tg[n,s]**4 - (6329.3 - 3.1619*(P*.98692)) + (14.3595 + 4.4552e-3*(P*0.98692))*model.Tg[n,s] - model.Tg[n,s]**2*(8.3395e-3 + 1.928e-6*P*0.98692) - 51.21 + 0.14215*P*0.98692)*1e-3
            #enth_rac = 2500
            Pr_g = model.CP[n,s]*(mu_g_i*1e-3)/nambda_g_i+.001
            Re_g = D_p*(model.m[n]/A)/(1-eps)/(mu_g_i*1e-3)+1
            U_gc = nambda_g_i/(pi*D_p**2)*(2+1.1*(Re_g+1e-5)**0.6*(Pr_g+1e-5)**0.33)*1e-3
            #U_gc = 1000
            return -model.tol <= enth_rac*1e3*(model.r_rac[n,s]/3600)/cp_c - U_gc*1e3/(1-eps)/rho_c/cp_c*(model.Tc[n,s]-model.Tg[n,s])
    model.heat_transfer4 = Constraint(model.nbed,model.seg,rule=heat_transfer_rule4)


    def sep_balance_rule(model,i):
        return model.m[n_bed-1]*model.z_mass[i,n_bed-1,n_seg-1]/MW_comp[i] == model.Fcyc*model.z_cyc[i]*(1+model.frac_purge) + model.Fp*model.z_prod[i]

    def z_cyc_rule1(model,i):
        if i < 2:
            return model.z_cyc[i]*P_sep/1.013 == H[i]*model.z_prod[i]
        else:
            return model.z_cyc[i]*P_sep == model.z_prod[i]*10**(4.8688-1113.928/(T_sep-10.409))

    def z_cyc_rule2(model):
        return sum(model.z_cyc[i] for i in NoComp) == 1
    
    def z_cyc_rule3(model):
        return sum(model.z_prod[i] for i in NoComp) == 1
    

    if _enable_recycle == True:
        model.sep_balance = Constraint(NoComp,rule=sep_balance_rule)
        model.z_cyc1 = Constraint(NoComp,rule=z_cyc_rule1)
        model.z_cyc2 = Constraint(rule=z_cyc_rule2)
        model.z_cyc3 = Constraint(rule=z_cyc_rule3)

    def z_mass_cyc_rule(model,i):
        return model.z_cyc[i]*MW_comp[i] == model.z_mass_cyc[i]*sum(model.z_cyc[i]*MW_comp[i] for i in NoComp)

    def m_cyc_rule(model):
        return model.Fcyc == model.mcyc*sum(model.z_mass_cyc[i]/MW_comp[i] for i in NoComp)
    if _enable_recycle == True:
        model.m_cyc = Constraint(rule=m_cyc_rule)
        model.zmass_cyc = Constraint(NoComp, rule=z_mass_cyc_rule)

    def _obj(model):
        if Fp_schedule:
            return (model.Fp - Fp_schedule)**2
        else:
            return model.Fp*model.z_prod[2]
    if Fp_schedule:
        model.obj = Objective(
            rule=_obj,
            sense = minimize)
    else:
        model.obj = Objective(
            rule=_obj,
            sense = maximize)
    

    opt = SolverFactory('ipopt')
    opt.options['max_iter']= 50000
    opt.options['halt_on_ampl_error']='yes'
    #opt.options['tol']= 1e-5
    result = opt.solve(model,tee=False)
    return {'model':model,'status': result.solver.status,'termination_condition':result.solver.termination_condition}
    



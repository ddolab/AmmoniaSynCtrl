import math
import numpy as np
import gym
from scipy.optimize import fsolve, root_scalar
from scipy.integrate import odeint,solve_ivp
import matplotlib
import matplotlib.pyplot as plt

    
class NH3reactor_ratio_T(gym.Env):

    def __init__(self,length,diameter,F_in,num_beds = 3,num_segs = 5, T_in = 313, P_in = 200, z_in = np.array([0.75,0.25,1e-10]),
    T0_init = 668, T_sep = 313.4, P_sep = 200, Time = 3000,init_temp=np.array([600,650,700]), ratios_init = np.array([0.9999,5e-5,5e-5]),
    dt = 0.1, interval = 60, full_obs = False, _enable_recycle = False, max_RR = 3,
    noise_Fin = None, noise_Tin = None, random_seed = 12345):

        self.L = length
        self.D = diameter
        self.nbed = num_beds
        self.nseg = num_segs
        self.init_temp = init_temp
        self.F_in = F_in
        self.z_in = z_in
        self.T_in = T_in
        self.P_in = P_in
        self.T0 = T0_init
        self.T0_init = T0_init
        self.Time = Time
        self.dt = dt
        self.discrete_steps = int(Time/interval)
        self.current_step = 0
        self.Tg = np.stack((init_temp[0]*np.ones(num_segs),
                            init_temp[1]*np.ones(num_segs),
                            init_temp[2]*np.ones(num_segs)))
        self.Tc = np.stack((init_temp[0]*np.ones(num_segs),
                            init_temp[1]*np.ones(num_segs),
                            init_temp[2]*np.ones(num_segs)))
        self.interval = interval
        self.sp_ratios = ratios_init
        self.sp_ratios_init = ratios_init
        self.Fcyc = F_in
        self.full_obs = full_obs
        self.acc_reward = 0
        self.reward_hist = []
        self.noise_Fin = noise_Fin
        self.noise_Tin = noise_Tin
        self.rng = np.random.default_rng(random_seed)
        self.enable_recycle = _enable_recycle
        self.recycle_stream = np.zeros(6)
        self.max_RR = max_RR
        self.purge_ratio = 0 
        self.T_sep = T_sep
        self.P_sep = P_sep
        self.r_rac = np.zeros([3,10])
        #self.observation_space = gym.spaces.Box(0, 1, shape = (3,3), dtype=np.float32)
        
        # for Dict full obs space
        if full_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "flowrates": gym.spaces.Box(0, 1000, shape=(self.nbed*self.nseg,), dtype=np.float32),
                    "temperatures": gym.spaces.Box(0, 1000, shape=(self.nbed*self.nseg,), dtype=np.float32),
                    "concentrations":gym.spaces.Box(0, 1, shape=(self.nbed*self.nseg,), dtype=np.float32)
                }
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    "flowrates": gym.spaces.Box(0, 1000, shape=(self.nbed,), dtype=np.float32),
                    "temperatures": gym.spaces.Box(0, 1000, shape=(self.nbed,), dtype=np.float32),
                    "concentrations":gym.spaces.Box(0, 1, shape=(self.nbed,), dtype=np.float32)
                }
            )
        if self.enable_recycle:
            self.action_space = gym.spaces.Dict(
                {
                "sp_ratios": gym.spaces.Box(low = 1e-8, high = 1, shape = (3,), dtype = np.float32),
                "T0": gym.spaces.Box(low = 500, high = 700, shape = (1,), dtype = np.float32),
                "Fcyc":gym.spaces.Box(low = 0, high = max_RR*F_in, shape = (1,), dtype = np.float32)
                }
            )
        else:
            self.action_space = gym.spaces.Dict(
                    {
                    "sp_ratios": gym.spaces.Box(low = 1e-8, high = 1, shape = (3,), dtype = np.float32),
                    "T0": gym.spaces.Box(low = 500, high = 700, shape = (1,), dtype = np.float32)
                    }
                )
   

    def Cp(self,T,P,x):
        C1 = np.array([33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974]) #H2
        C2 = np.array([19.50583, 19.88705, -8.598535, 1.369784, .527601, -4.935202, 212.39])#N2
        C3 = np.array([19.99563, 49.77119, -15.37599, 1.921168, .189174, -53.30667, -45.89806])#NH3
        CP_comp = np.zeros(3)
        C = np.array([C1,C2,C3])
        for i in range(2):
            CP_comp[i] = (C[i,0] + C[i,1]*(T/1000) + C[i,2]*(T/1000)**2 + C[i,3]*(T/1000)**3 + C[i,4]/(T/1000)**2)
        CP_comp[2] = 4.184*(6.5846 - 0.61251e-2*T + 0.23663e-5*T**2 - 1.5981e-9*T**3 + (96.1678-0.067571*P*0.98692) + (-.2225 + 1.6847e-4*P*0.98692)*T + (1.289e-4 - 1.0095e-7*P*.98692)*T**2)
        return np.dot(x,CP_comp)

    def mix_process(self, u1, u2):
        # parse the inputs
        F1 = u1[0]
        T1 = u1[1]
        P1 = u1[2]
        z1 = u1[3:6]
        F2 = u2[0]
        T2 = u2[1]
        P2 = u2[2]
        z2 = u2[3:6]

        #calculate flowrate, pressure, temperature, molar fraction after mixing
        CP1 = self.Cp(T1,P1,z1)
        CP2 = self.Cp(T2,P2,z2)
        F = F1 + F2
        T = (F1*CP1*T1+F2*CP2*T2)/(F1*CP1+F2*CP2)
        P = (P1+P2)/2
        z = (F1*z1+F2*z2)/(F1+F2)
        return np.array([F,T,P,z[0],z[1],z[2]])

    def unit_cal_spatial(self,x,u,d_o,len):
        MW_comp = np.array([2, 28.01, 17.03])*1e-3 #molecular weight
        R       = 8.314
        D_p     = .00285 #catalyst particle diameter
        rho_c   = 2200 #catalyst denstiy 2200 kg/m3
        eps      = .33 #bed void fraction
        mu_g_i  = 0.028 #cP viscosity
        
        # parse the inputs
        F_g_i = u[0]
        T_g_i = u[1]
        P_g_i = u[2]
        z_g_i = u[3:6]
        #dimension
        d_i = d_o - .04
        area = d_i**2*math.pi/4

        #properties
        MW_g_i = np.dot(MW_comp,z_g_i)
        m_g_i = F_g_i*MW_g_i
        rho_g_i = P_g_i*1e5/R/T_g_i # assume ideal gas
        z_mass_i = z_g_i[2]*MW_comp[2]/MW_g_i

        #states
        T_g_o = x[0]
        T_c = x[1]

        #kinetics
        k1 = 1.79e4 * math.exp(-87090/(R*T_c))
        k2 = 2.57e16 * math.exp(-198464/(R*T_c))
        P_g_o = P_g_i - m_g_i/(rho_g_i*MW_g_i)/area/D_p * (1-eps)/eps**3*(150*(1-eps)*(mu_g_i*1e-3)/D_p + 1.75*m_g_i/area)*len/1e5
        #print(P_g_o)
        def mass_balance(x):
            MW_g = np.dot(MW_comp,x[1:4])
            z_mass = x[3]*MW_comp[2]/MW_g
            p_g_p = P_g_i*x[1:4]
            r_rac = (k1*p_g_p[1]*(p_g_p[0]+1e-5)**1.5/(p_g_p[2]+1e-5) - k2*p_g_p[2]/(p_g_p[0]+1e-5)**1.5)*34/rho_c*4.75 # Ref [Morud and Skogestad 1998] [kg NH3/kg cat/hr]
            return  [- m_g_i/eps/area*((z_mass - z_mass_i)/(len)) + (1-eps)*rho_c/eps*(r_rac/3600),
                    x[1] + x[2] + x[3] - 1,
                    x[0]*x[2] - F_g_i*z_g_i[1] + (x[0]*x[3] - F_g_i*z_g_i[2])/2,
                    x[0]*x[1] - F_g_i*z_g_i[0] + (x[0]*x[3] - F_g_i*z_g_i[2])/2*3]

        x0 = np.insert(z_g_i,0,F_g_i)
        mb = fsolve(mass_balance, x0)
        p_g_p = P_g_i*z_g_i
        r_rac = (k1*p_g_p[1]*(p_g_p[0]+1e-5)**1.5/(p_g_p[2]+1e-5) - k2*p_g_p[2]/(p_g_p[0]+1e-5)**1.5)*34/rho_c*4.75
        F_g_o = mb[0]
        z_g_o = mb[1:4]
        y = np.array([F_g_o, T_g_o, P_g_o])
        y = np.append(y,z_g_o)
        return y, r_rac

    def unit_cal_time(self,t,x,u,y,d_o,len):

        MW_comp = np.array([2, 28.01, 17.03])*1e-3 #molecular weight
        R       = 8.314
        D_p     = .00285 #catalyst particle diameter
        cp_c    = 1100 #catalyst heat capacity J/kg/K
        rho_c   = 2200 #catalyst denstiy 2200 kg/m3
        eps      = .33 #bed void fraction
        mu_g_i  = 0.028 #cP viscosity
        nambda_g_i = 0.231 #W/mK thermal conductivity
        
        # parse the inputs
        F_g_i = u[0]
        T_g_i = u[1]
        P_g_i = u[2]
        z_g_i = u[3:6]
        #F_g_o = y[0]
        #T_g_o = y[1]
        P_g_o = y[2]
        z_g_o = y[3:6]
        #dimension
        d_i = d_o - .04
        area = d_i**2*math.pi/4

        #properties
        MW_g_i = np.dot(MW_comp,z_g_i)
        m_g_i = F_g_i*MW_g_i

        #states
        T_g_o = x[0]
        T_c = x[1]

        #kinetics
        k1 = 1.79e4 * math.exp(-87090/(R*T_c))
        k2 = 2.57e16 * math.exp(-198464/(R*T_c))
        #print(T_g_o)


        MW_g = np.dot(MW_comp,z_g_o)
        rho_g = P_g_i*1e5/R/T_g_o
  
        p_g_p = P_g_i*z_g_o
        enth_rac = -4.184/MW_comp[2]*(-9184 - 7.2949*T_g_o + 0.34996e-2*(T_g_o)**2 + 0.03356e-5*(T_g_o)**3 - 0.11625e-9*(T_g_o)**4 - (6329.3 - 3.1619*(P_g_i*.98692)) + (14.3595 + 4.4552e-3*(P_g_i*0.98692))*(T_g_o) - (T_g_o)**2*(8.3395e-3 + 1.928e-6*P_g_i*0.98692) - 51.21 + 0.14215*P_g_i*0.98692)*1e-3
        r_rac = (k1*p_g_p[1]*(p_g_p[0]+1e-5)**1.5/(p_g_p[2]+1e-5) - k2*p_g_p[2]/(p_g_p[0]+1e-5)**1.5)*34/rho_c*4.75 #Ref [Morud and Skogestad 1998] [kg NH3/kg cat/hr]
        
        #Heat Transfer
        CP = self.Cp(T_g_o, P_g_o, z_g_o)
        Pr_g = CP/MW_g*(mu_g_i*1e-3)/nambda_g_i+.001
        Re_g = D_p*(m_g_i/area)/(1-eps)/(mu_g_i*1e-3)+1
        U_gc = nambda_g_i/(math.pi*D_p**2)*(2+1.1*(Re_g+1e-5)**.6*(Pr_g+1e-5)**.33)*1e-3
     
        #deriverive terms and states

        dx = [- m_g_i/MW_g/eps/rho_g/area*((T_g_o - T_g_i)/len) + U_gc*1e3/eps/rho_g/CP*(T_c-T_g_o),
        enth_rac*1e3*(r_rac/3600)/cp_c - U_gc*1e3/(1-eps)/rho_c/cp_c*(T_c-T_g_o)]

        return dx

    def bed_cal(self,bed_index,input, _full_obs = False, _with_updating_dx = True):
            u = input
            obs = {
                "flowrate":np.array([]),
                "temperature":np.array([]),
                "pressure":np.array([]),
                "x1":np.array([]),
                "x2":np.array([]),
                "x3":np.array([])
            }
            
            for i in range(self.nseg):
                if _full_obs:
                    obs["flowrate"] = np.append(obs["flowrate"], u[0])
                    obs["temperature"] = np.append(obs["temperature"], u[1])
                    obs["pressure"] = np.append(obs["pressure"], u[2])
                    obs["x1"] = np.append(obs["x1"], u[3])
                    obs["x2"] = np.append(obs["x2"], u[4])
                    obs["x3"] = np.append(obs["x3"], u[5])
                x = np.array([self.Tg[bed_index,i] , self.Tc[bed_index,i]])
                len = []
                for l in self.L:
                    len.append(l/self.nseg)
                y, self.r_rac[bed_index,i] = self.unit_cal_spatial(x,u,self.D,len[bed_index])
                if _with_updating_dx:
                    #t = np.linspace(0, self.dt, 3)
                    t = [0, self.dt]
                    sol= solve_ivp(self.unit_cal_time, t, x, 
                                    method = 'LSODA',
                                    args=(u,y,self.D,len))
                    self.Tg[bed_index,i] = sol.y[0,-1]
                    self.Tc[bed_index,i] = sol.y[1,-1]
                u = y
            return u, obs

    def recycle_cal(self,u,Fcyc,P_sep=200,T_sep = 313.34):
        
        K = 1/P_sep*10**(4.8688-1113.928/(T_sep-10.409))
        F_cyc_ammonia = K*u[0]*u[5]
        Fp = (1-K)*u[0]*u[5]
        F_cyc = F_cyc_ammonia + u[0]*(u[3]+u[4])
        z1 = u[3]*u[0]/F_cyc
        z2 = u[4]*u[0]/F_cyc
        z3 = 1-z1-z2
        y = np.array([min(F_cyc,Fcyc), T_sep,P_sep,z1,z2,z3])
        return y, Fp

    # for discrete action space only
    def action_mapping(self,action):
        T0 = self.T0
        sp_ratios = np.zeros(3)
        sp_ratios[0] = action["sp_ratios"][0]/sum(action["sp_ratios"])
        sp_ratios[1] = action["sp_ratios"][1]/sum(action["sp_ratios"])
        sp_ratios[2] = action["sp_ratios"][2]/sum(action["sp_ratios"])
        T0 = action["T0"]
        if self.enable_recycle:
            Fcyc = action["Fcyc"]
            return sp_ratios, T0, Fcyc
        return sp_ratios, T0       
    
    def _env_step(self,action = None, _is_reset = False, _full_obs = False):
        
        if not _is_reset:
            if self.enable_recycle:
                self.sp_ratios , self.T0, self.Fcyc = action
            else:
                self.sp_ratios , self.T0 = action
        
        y = np.zeros(6)
        #States = np.zeros([self.nbed,3])
        States = {
                "flowrates": np.array([]),
                "temperatures": np.array([]),
                "concentrations":np.array([])
        }
        Reward = 0
        n_steps = int(self.interval/self.dt)
        with_updating_dx = True
        for k in range(n_steps + 1):
            
            if self.noise_Fin:
                F_in = self.F_in + self.rng.normal(self.noise_Fin[0], self.noise_Fin[1])
            else:
                F_in = self.F_in
            feed_stream = np.array([F_in,self.T_in,self.P_in,self.z_in[0],self.z_in[1],self.z_in[2]])
            if self.recycle_stream[0] != 0:
                feed_stream = self.mix_process(feed_stream, self.recycle_stream)
            if self.noise_Tin:
                T0 = self.T0 + self.rng.normal(self.noise_Tin[0], self.noise_Tin[1])
            else:
                T0 = self.T0
            F = feed_stream[0] * self.sp_ratios
            if k == n_steps:
                with_updating_dx = False
            for i in range(self.nbed):
                if i == 0:
                    input = np.array([F[0],T0,feed_stream[2],feed_stream[3],feed_stream[4],feed_stream[5]])
                else:
                    input = self.mix_process(np.array([F[i],feed_stream[1],feed_stream[2],feed_stream[3],feed_stream[4],feed_stream[5]]),y)
                y,obs = self.bed_cal(bed_index=i,input=input, _full_obs = _full_obs, _with_updating_dx = with_updating_dx)
                product = y[0]*y[5]
                #pressure = y[2]
                temperature = y[1]
                if k == n_steps:
                    
                    if not _full_obs:
                        
                        States["flowrates"] = np.append(States["flowrates"], y[0])
                        States["temperatures"] = np.append(States["temperatures"], y[1])
                        States["concentrations"] = np.append(States["concentrations"], y[5])
                    else:
                        States["flowrates"] = np.concatenate((States["flowrates"], obs["flowrate"]))
                        States["temperatures"] = np.concatenate((States["temperatures"], obs["temperature"]))
                        States["concentrations"] = np.concatenate((States["concentrations"], obs["x3"]))                    
                if i == self.nbed - 1:
                    
                    if self.enable_recycle:
                        self.recycle_stream, product = self.recycle_cal(y,self.Fcyc,P_sep = self.P_sep, T_sep = self.T_sep)
                    if k == 0 or k == n_steps:
                        Reward = Reward + product*self.dt/2
                    else:
                        Reward = Reward + product*self.dt
        return States, Reward

    def step(self,action,MPC_purpose = False):
        
        States, Reward = self._env_step(action = self.action_mapping(action), _full_obs = self.full_obs)
        self.current_step += 1
        self.acc_reward += Reward
        if self.current_step >= self.discrete_steps:
            terminated = True
        else:
            terminated = False
        if math.isnan(Reward):
            truncated = True
        else:
            truncated = False
        info = {"step ": self.current_step , " reward " : Reward}
        if MPC_purpose:
            return self.Tg, self.Tc, Reward, terminated
        else:
            return States, Reward, terminated, info

    def reset(self, seed = None, options = []):
        #super().reset(seed=seed)
        #global reward_hist
        self.current_step = 0
        self.Tg = np.stack((self.init_temp[0]*np.ones(self.nseg),
                            self.init_temp[1]*np.ones(self.nseg),
                            self.init_temp[2]*np.ones(self.nseg)))
        self.Tc = np.stack((self.init_temp[0]*np.ones(self.nseg),
                            self.init_temp[1]*np.ones(self.nseg),
                            self.init_temp[2]*np.ones(self.nseg)))
        self.sp_ratios = self.sp_ratios_init
        self.T0 = self.T0_init
        self.Fcyc = self.F_in
        self.recycle_stream = np.zeros(6)
        x_init, _ = self._env_step(_is_reset = True, _full_obs = self.full_obs)
        #info = "reset complete"
        #print("accumulated reward", self.acc_reward)
        '''
        if self.acc_reward != 0:
            self.reward_hist = np.append(self.reward_hist, self.acc_reward)
            display.clear_output(wait=True)
            plt.plot(self.reward_hist)
            plt.plot(np.cumsum(self.reward_hist)/np.arange(1,self.reward_hist.size+1))
            plt.xlabel('Best: %g at epsiode %d' % (np.max(self.reward_hist),np.argmax(self.reward_hist)))
            display.display(plt.gcf())
        self.acc_reward = 0
        '''
        return x_init
        
    def get_reward_hist(self):
        
        return self.reward_hist
    

    
class NH3reactor(gym.Env):


    def __init__(self,length,diameter,F_in, HX_A = 61, HX_U = 536,num_beds = 3,num_segs = 10, T_in = 313, P_in = 200, z_in = np.array([0.75,0.25,1e-10]),
    T_sep = 313.4, P_sep = 196, V_sep = 50, liquid_frac_sep = 0.3, Time = 3000,init_Tg=700*np.ones([3,10]), init_Tc=700*np.ones([3,10]),ratios_init = np.array([0.5,0.5,5e-5,5e-5]),
    dt = 0.1, interval = 60, _enable_recycle = False, max_RR = 3, z_prod_init = 0.975,
    noise_Fin = None, noise_Tin = None, noise_Tsep = None, noise_type = 'normal', random_seed = 12345):

        self.L = length
        self.D = diameter
        self.nbed = num_beds
        self.nseg = num_segs
        self.HX_A = HX_A
        self.HX_U = HX_U
        self.init_Tg = np.copy(init_Tg)
        self.init_Tc = np.copy(init_Tc)
        self.Tbed_0 = np.zeros(num_beds)
        self.F_in = F_in
        self.Fp = F_in[0]/2
        self.feed_flowrate = F_in[0]
        self.z_in = z_in
        self.z_prod_init = z_prod_init
        self.T_in = T_in
        self.P_in = P_in
        self.Time = Time
        self.dt = dt
        self.discrete_steps = int(Time/interval)
        self.current_step = 0
        self.Tg = np.copy(init_Tg)
        self.Tc = np.copy(init_Tc)
        self.F = np.zeros([self.nbed,self.nseg])
        self.z = np.zeros([self.nbed,self.nseg,3])
        self.interval = interval
        self.sp_ratios = ratios_init
        self.sp_ratios_init = ratios_init
        self.Fcyc = F_in[0]
        self.acc_reward = 0
        self.reward_hist = []
        self.noise_Fin = noise_Fin
        self.noise_Tin = noise_Tin
        self.noise_Tsep = noise_Tsep
        self.noise_type = noise_type
        self.rng = np.random.default_rng(random_seed)
        self.enable_recycle = _enable_recycle
        self.recycle_stream = np.zeros(6)
        self.max_RR = max_RR
        self.T_sep = T_sep
        self.P_sep = P_sep
        self.V_sep = V_sep
        self.Nv_sep = P_sep*1e5*V_sep*(1-liquid_frac_sep)/8.314/T_sep
        self.Nl_sep = V_sep*liquid_frac_sep*40080
        self.N_sep = [self.Nv_sep*0.75, self.Nv_sep*0.25, self.Nl_sep]
        self.N_sep,self.z_cyc, self.z_prod = self.init_sep()
        self.purge_ratio = 0
        self.r_rac = np.zeros([3,10])
        self.dist_rac = np.zeros([3,10])
        #self.observation_space = gym.spaces.Box(0, 1, shape = (3,3), dtype=np.float32)
        
        # for Dict full obs space

        self.observation_space = gym.spaces.Dict(
            {
                "feed_flowrates": gym.spaces.Box(0, 1000, shape=(self.nbed,), dtype=np.float32),
                "flowrates": gym.spaces.Box(0, 1000, shape=(self.nbed,), dtype=np.float32),
                "temperatures": gym.spaces.Box(0, 1000, shape=(self.nbed,), dtype=np.float32),
                "concentrations":gym.spaces.Box(0, 1, shape=(self.nbed,), dtype=np.float32)
            }
        )

        if self.enable_recycle:
            self.action_space = gym.spaces.Box(low = 0.01, high = 1, shape = (6,), dtype = np.float32)
        else:
            self.action_space = gym.spaces.Box(low = 0.01, high = 1, shape = (5,), dtype = np.float32)

    def sep_cal(self,x, T_sep):
        # x0: Nv
        # x1: Nl
        # x2~x4: z_cyc
        # x5~x7: z_prod
        K = 1/self.P_sep*10**(4.8688-1113.928/(T_sep-10.409))
        Henry_const = np.array([[-3.68607, -2.29337],
                        [0.596736*1e4, 0.5294740*1e4],
                        [-0.642828*1e6,-0.521881*1e6]])
        H = np.exp(np.matmul(Henry_const.transpose(),np.array([1,1/T_sep,1/T_sep**2])))
        return [
            x[0]*x[2] + x[1]*x[5] - self.N_sep[0],
            x[0]*x[3] + x[1]*x[6] - self.N_sep[1],
            x[0]*x[4] + x[1]*x[7] - self.N_sep[2],
            x[2]*self.P_sep/1.013 - H[0]*x[5],
            x[3]*self.P_sep/1.013 - H[1]*x[6],
            x[4] - K*x[7],
            x[2]+x[3]+x[4]-1,
            x[5]+x[6]+x[7]-1]  
    
    def sep_cal_init(self,x, T_sep):
        # x0~x2: N_sep
        # x3~x5: z_cyc
        # x6~x8: z_prod
        K = 1/self.P_sep*10**(4.8688-1113.928/(T_sep-10.409))
        Henry_const = np.array([[-3.68607, -2.29337],
                        [0.596736*1e4, 0.5294740*1e4],
                        [-0.642828*1e6,-0.521881*1e6]])
        H = np.exp(np.matmul(Henry_const.transpose(),np.array([1,1/T_sep,1/T_sep**2])))
        return [
            self.Nv_sep*x[3] + self.Nl_sep*x[6] - x[0],
            self.Nv_sep*x[4] + self.Nl_sep*x[7] - x[1],
            self.Nv_sep*x[5] + self.Nl_sep*x[8] - x[2],
            x[8] - self.z_prod_init,
            x[3]*self.P_sep/1.013 - H[0]*x[6],
            x[4]*self.P_sep/1.013 - H[1]*x[7],
            x[5] - K*x[8],
            x[3]+x[4]+x[5]-1,
            x[6]+x[7]+x[8]-1] 
    
    def init_sep(self):
        x0 = self.N_sep + [0.69,0.23,0.076,0.02,0.005,0.975]
        result = fsolve(self.sep_cal_init, x0, args = (self.T_sep))
        N_sep = list(result[:3])
        z_cyc = list(result[3:6])
        z_prod = list(result[6:9])
        return N_sep,z_cyc,z_prod
    
    def Cp(self,T,P,x):
        C1 = np.array([33.066178, -11.363417, 11.432816, -2.772874, -0.158558, -9.980797, 172.707974]) #H2
        C2 = np.array([19.50583, 19.88705, -8.598535, 1.369784, .527601, -4.935202, 212.39])#N2
        C3 = np.array([19.99563, 49.77119, -15.37599, 1.921168, .189174, -53.30667, -45.89806])#NH3
        CP_comp = np.zeros(3)
        C = np.array([C1,C2,C3])
        for i in range(2):
            CP_comp[i] = (C[i,0] + C[i,1]*(T/1000) + C[i,2]*(T/1000)**2 + C[i,3]*(T/1000)**3 + C[i,4]/(T/1000)**2)
        CP_comp[2] = 4.184*(6.5846 - 0.61251e-2*T + 0.23663e-5*T**2 - 1.5981e-9*T**3 + (96.1678-0.067571*P*0.98692) + (-.2225 + 1.6847e-4*P*0.98692)*T + (1.289e-4 - 1.0095e-7*P*.98692)*T**2)
        return np.dot(x,CP_comp)

    def mix_process(self, u1, u2):
        # parse the inputs
        F1 = u1[0]
        T1 = u1[1]
        P1 = u1[2]
        z1 = u1[3:6]
        F2 = u2[0]
        T2 = u2[1]
        P2 = u2[2]
        z2 = u2[3:6]

        #calculate flowrate, pressure, temperature, molar fraction after mixing
        CP1 = self.Cp(T1,P1,z1)
        CP2 = self.Cp(T2,P2,z2)
        F = F1 + F2
        T = (F1*CP1*T1+F2*CP2*T2)/(F1*CP1+F2*CP2)
        P = min(P1,P2)
        z = (F1*z1+F2*z2)/(F1+F2)
        return np.array([F,T,P,z[0],z[1],z[2]])
    
    def heat_exchanger(self, U, A, Tcold_in, Thot_in,Fcold,Fhot,CPcold,CPhot):

        Cmin = min(Fcold*CPcold,Fhot*CPhot)
        Cmax = max(Fcold*CPcold,Fhot*CPhot)
        Cr = Cmin/Cmax
        qmax = Cmin*(Thot_in-Tcold_in)
        NTU = U*A/Cmin
        if Cr == 1:
            epsilon = NTU/(1+NTU)
        else:
            try:
                epsilon = (1-math.exp(-NTU*(1-Cr)))/(1-Cr*math.exp(-NTU*(1-Cr)))
            except:
                epsilon = NTU/(1+NTU)
        q = qmax * epsilon
        Tcold_out = q/(Fcold*CPcold) + Tcold_in
        Thot_out = -q/(Fhot*CPhot) + Thot_in
        #print(Tcold_out)0
        return Tcold_out, Thot_out
    
    def unit_cal_spatial(self,x,u,d_o,len):
        MW_comp = np.array([2, 28.01, 17.03])*1e-3 #molecular weight
        R       = 8.314
        D_p     = .00285 #catalyst particle diameter
        rho_c   = 2200 #catalyst denstiy 2200 kg/m3
        eps      = .33 #bed void fraction
        mu_g_i  = 0.028 #cP viscosity
        
        # parse the inputs
        F_g_i = u[0]
        T_g_i = u[1]
        P_g_i = u[2]
        z_g_i = u[3:6]
        #dimension
        d_i = d_o 
        area = d_i**2*math.pi/4

        #properties
        MW_g_i = np.dot(MW_comp,z_g_i)
        m_g_i = F_g_i*MW_g_i
        rho_g_i = P_g_i*1e5/R/T_g_i # assume ideal gas
        z_mass_i = z_g_i[2]*MW_comp[2]/MW_g_i

        #states
        T_g_o = x[0]
        T_c = x[1]

        #kinetics
        k1 = 1.79e4 * math.exp(-87090/(R*T_c))
        k2 = 2.57e16 * math.exp(-198464/(R*T_c))
        P_g_o = P_g_i - m_g_i/(rho_g_i*MW_g_i)/area/D_p * (1-eps)/eps**3*(150*(1-eps)*(mu_g_i*1e-3)/D_p + 1.75*m_g_i/area)*len/1e5
        #P_g_o = P_g_i
        def mass_balance(x):
            MW_g = np.dot(MW_comp,x[1:4])
            z_mass = x[3]*MW_comp[2]/MW_g
            p_g_p = P_g_i*x[1:4]
            r_rac = (k1*p_g_p[1]*(p_g_p[0]+1e-5)**1.5/(p_g_p[2]+1e-5) - k2*p_g_p[2]/(p_g_p[0]+1e-5)**1.5)*34/rho_c*4.75 # Ref [Morud and Skogestad 1998] [kg NH3/kg cat/hr]
            return  [- m_g_i/eps/area*((z_mass - z_mass_i)/(len)) + (1-eps)*rho_c/eps*(r_rac/3600),
                    x[1] + x[2] + x[3] - 1,
                    x[0]*x[2] - F_g_i*z_g_i[1] + (x[0]*x[3] - F_g_i*z_g_i[2])/2,
                    x[0]*x[1] - F_g_i*z_g_i[0] + (x[0]*x[3] - F_g_i*z_g_i[2])/2*3]
        p_g_p = P_g_i*z_g_i
        r_rac = (k1*p_g_p[1]*(p_g_p[0]+1e-5)**1.5/(p_g_p[2]+1e-5) - k2*p_g_p[2]/(p_g_p[0]+1e-5)**1.5)*34/rho_c*4.75
        #add disturbance to reaction rate
        dist_rac = self.rng.normal(0, 0*r_rac)
        r_rac = r_rac + dist_rac
        x0 = np.insert(z_g_i,0,F_g_i)
        mb = fsolve(mass_balance, x0)
        F_g_o = mb[0]
        z_g_o = mb[1:4]
        y = np.array([F_g_o, T_g_o, P_g_o])
        y = np.append(y,z_g_o)
        return y, r_rac, dist_rac

    def unit_cal_time(self,t,x,u,y,d_o,len,dist_rac):

        MW_comp = np.array([2, 28.01, 17.03])*1e-3 #molecular weight
        R       = 8.314
        D_p     = .00285 #catalyst particle diameter
        cp_c    = 1100 #catalyst heat capacity J/kg/K
        rho_c   = 2200 #catalyst denstiy 2200 kg/m3
        eps      = .33 #bed void fraction
        mu_g_i  = 0.028 #cP viscosity
        nambda_g_i = 0.231 #W/mK thermal conductivity
        
        # parse the inputs
        F_g_i = u[0]
        T_g_i = u[1]
        P_g_i = u[2]
        z_g_i = u[3:6]
        #F_g_o = y[0]
        #T_g_o = y[1]
        P_g_o = y[2]
        z_g_o = y[3:6]
        #dimension
        d_i = d_o
        area = d_i**2*math.pi/4

        #properties
        MW_g_i = np.dot(MW_comp,z_g_i)
        m_g_i = F_g_i*MW_g_i

        #states
        T_g_o = x[0]
        T_c = x[1]

        #kinetics
        k1 = 1.79e4 * math.exp(-87090/(R*T_c))
        k2 = 2.57e16 * math.exp(-198464/(R*T_c))
        #print(T_g_o)


        MW_g = np.dot(MW_comp,z_g_o)
        rho_g = P_g_i*1e5/R/T_g_o
  
        p_g_p = P_g_i*z_g_o
        enth_rac = -4.184/MW_comp[2]*(-9184 - 7.2949*T_g_o + 0.34996e-2*(T_g_o)**2 + 0.03356e-5*(T_g_o)**3 - 0.11625e-9*(T_g_o)**4 - (6329.3 - 3.1619*(P_g_i*.98692)) + (14.3595 + 4.4552e-3*(P_g_i*0.98692))*(T_g_o) - (T_g_o)**2*(8.3395e-3 + 1.928e-6*P_g_i*0.98692) - 51.21 + 0.14215*P_g_i*0.98692)*1e-3
        r_rac = (k1*p_g_p[1]*(p_g_p[0]+1e-5)**1.5/(p_g_p[2]+1e-5) - k2*p_g_p[2]/(p_g_p[0]+1e-5)**1.5)*34/rho_c*4.75 #Ref [Morud and Skogestad 1998] [kg NH3/kg cat/hr]
        #add disturbance to reaction rate
        r_rac = r_rac + dist_rac
        #Heat Transfer
        CP = self.Cp(T_g_o, P_g_o, z_g_o)
        Pr_g = CP/MW_g*(mu_g_i*1e-3)/nambda_g_i+.001
        Re_g = D_p*(m_g_i/area)/(1-eps)/(mu_g_i*1e-3)+1
        U_gc = nambda_g_i/(math.pi*D_p**2)*(2+1.1*(Re_g+1e-5)**.6*(Pr_g+1e-5)**.33)*1e-3
     
        #deriverive terms and states

        dx = [- m_g_i/MW_g/eps/rho_g/area*((T_g_o - T_g_i)/len) + U_gc*1e3/eps/rho_g/CP*(T_c-T_g_o),
        enth_rac*1e3*(r_rac/3600)/cp_c - U_gc*1e3/(1-eps)/rho_c/cp_c*(T_c-T_g_o)]

        return dx

    def recycle_cal(self,u,Fcyc,Fp, T_sep, dt):
       
        for i in range(3):
            self.N_sep[i] = self.N_sep[i] + (u[0]*u[i+3] - Fp*self.z_prod[i] - Fcyc*self.z_cyc[i]*(1+self.purge_ratio))*dt
        x0 = [self.Nv_sep,self.Nl_sep] + self.z_cyc + self.z_prod
        result = fsolve(self.sep_cal, x0, args = (T_sep))
        Nv = result[0]
        Nl = result[1]
        z_cyc = list(result[2:5])
        z_prod = list(result[5:8])
        self.Nv_sep = Nv
        self.Nl_sep = Nl
        self.z_cyc = z_cyc
        self.z_prod = z_prod
        self.P_sep = Nv/1e5/(self.V_sep-Nl/40080)*8.314*T_sep
        y = np.array([Fcyc,self.T_in,200,self.z_cyc[0],self.z_cyc[1],self.z_cyc[2]])
        return y
    

    def bed_cal(self,bed_index,input, _with_updating_dx = True):
            u = input
            
            for i in range(self.nseg):

                x = np.array([self.Tg[bed_index,i] , self.Tc[bed_index,i]])
                len = []
                for l in self.L:
                    len.append(l/self.nseg)
                y , self.r_rac[bed_index,i], self.dist_rac[bed_index,i] = self.unit_cal_spatial(x,u,self.D,len[bed_index])
                self.F[bed_index,i] = y[0]
                for comp in range(3):
                    self.z[bed_index,i,comp] = y[comp+3]
                if _with_updating_dx:
                    #t = np.linspace(0, self.dt, 3)
                    t = [0, self.dt]
                    sol= solve_ivp(self.unit_cal_time, t, x, 
                                    method = 'LSODA',
                                    args=(u,y,self.D,len[bed_index],
                                          self.dist_rac[bed_index,i]))
                    self.Tg[bed_index,i] = sol.y[0,-1]
                    self.Tc[bed_index,i] = sol.y[1,-1]
                u = y
            #print(self.Tg)
            return u

    # for discrete action space only
    def action_mapping(self,action):
        sp_ratios = np.zeros(4)
        sp_ratios[0] = action[0]/(action[0]+action[1]+action[2]+action[3])
        sp_ratios[1] = action[1]/(action[0]+action[1]+action[2]+action[3])
        sp_ratios[2] = action[2]/(action[0]+action[1]+action[2]+action[3])
        sp_ratios[3] = action[3]/(action[0]+action[1]+action[2]+action[3])
        feed_flowrate = action[4]*2/self.feed_flowrate
        if self.enable_recycle:
            Fcyc = action[5]*self.F_in*self.max_RR
            Fp = action[6]*self.F_in/2
            frac_purge = action[7]
            return sp_ratios, feed_flowrate, Fcyc, Fp, frac_purge  
        return sp_ratios, feed_flowrate
    
    def _get_noise(self):
        F_noise = self.noise_Fin*self.feed_flowrate
        T_noise = self.noise_Tin*self.T_in
        Tsep_noise = self.noise_Tsep*self.T_sep
        if self.noise_type == 'normal':
            return self.rng.normal(0, F_noise), self.rng.normal(0, T_noise), self.rng.normal(0, Tsep_noise)
        elif self.noise_type == 'uniform':
            return self.rng.uniform(low = -F_noise, high = F_noise), 
        self.rng.uniform(low = -T_noise, high = T_noise), self.rng.uniform(low = -Tsep_noise, high = Tsep_noise)
        
    def _env_step(self,action = None, _is_reset = False):
        
        Reward = 0
        if not _is_reset:
            if self.enable_recycle:
                self.sp_ratios , self.feed_flowrate, self.Fcyc, self.Fp, self.purge_ratio = action
            else:
                self.sp_ratios, self.feed_flowrate = action
        y = np.zeros(6)

        #States = np.zeros([self.nbed,3])
        States = {
                "feed_flowrates": np.array([]),
                "flowrates": np.array([]),
                "temperatures": np.array([]),
                "concentrations":np.array([])
        }
        n_steps = int(self.interval/self.dt)
        with_updating_dx = True
        for k in range(n_steps + 1):
            Fin_noise, Tin_noise, Tsep_noise = self._get_noise()
            F_in = self.feed_flowrate + Fin_noise
            T_in = self.T_in + Tin_noise
            T_sep = self.T_sep + Tsep_noise
            feed_stream = np.array([F_in,T_in,self.P_in,self.z_in[0],self.z_in[1],self.z_in[2]])
            if self.current_step == 0 and k == 0:
                self.recycle_stream = self.recycle_cal(feed_stream,self.Fcyc,self.Fp, T_sep, dt = 0)
            if self.enable_recycle:
                feed_stream = self.mix_process(feed_stream, self.recycle_stream)
            F = feed_stream[0] * self.sp_ratios
            #pre_heating:
            if k == 0:
                Thot_in = self.Tg[-1,-1]
                Fhot = feed_stream[0]
                CPhot = self.Cp(Thot_in,self.P_in,self.z_in)
            else:
                Thot_in = y[1]
                Fhot = y[0]
                CPhot = self.Cp(Thot_in,self.P_in,y[3:6])
            CPcold = self.Cp(feed_stream[1],feed_stream[2],feed_stream[3:6])
            T0, _ = self.heat_exchanger(self.HX_U, self.HX_A, 
                                    Tcold_in=feed_stream[1], Thot_in=Thot_in,
                                    Fhot = Fhot, Fcold = F[0],
                                    CPcold=CPcold, CPhot=CPhot)
            if k == n_steps:
                with_updating_dx = False
            for i in range(self.nbed):
                if i == 0:
                    input = self.mix_process(np.array([F[0],T0,feed_stream[2],feed_stream[3],feed_stream[4],feed_stream[5]]),
                                             np.array([F[1],feed_stream[1],feed_stream[2],feed_stream[3],feed_stream[4],feed_stream[5]]))
                else:
                    input = self.mix_process(np.array([F[i+1],feed_stream[1],feed_stream[2],feed_stream[3],feed_stream[4],feed_stream[5]]),y)
                self.Tbed_0[i] = input[1]
                y = self.bed_cal(bed_index=i,input=input.copy(), _with_updating_dx = with_updating_dx)
                #product = y[0]*y[5]
                if k == n_steps:
                        States["feed_flowrates"] = np.append(States["feed_flowrates"], input[0])
                        States["flowrates"] = np.append(States["flowrates"], y[0])
                        States["temperatures"] = np.append(States["temperatures"], y[1])
                        States["concentrations"] = np.append(States["concentrations"], y[5])
                if i == self.nbed - 1:
                    if k == 0 or k == n_steps:
                        Reward = Reward + self.Fp*self.dt/2
                        if self.enable_recycle:
                            self.recycle_stream = self.recycle_cal(y,self.Fcyc,self.Fp, T_sep, dt = self.dt/2)
                    else:
                        Reward = Reward + self.Fp*self.dt
                        if self.enable_recycle:
                            self.recycle_stream = self.recycle_cal(y,self.Fcyc,self.Fp, T_sep, dt = self.dt)
        #print(States["concentrations"])
        return States, Reward, y

    def step(self,action,MPC_purpose = False):
        #steps_per_feed = int(self.discrete_steps/len(self.F_in)) #how many steps for each feed flowrate
        self.feed_flowrate = self.F_in[self.current_step] #feed flowrate from schedule
        if MPC_purpose:
            action[0] = np.clip(action[0], 0, 1)
            States, Reward,y = self._env_step(action = action)
        else:
            States, Reward,y = self._env_step(action = self.action_mapping(action))
        self.current_step += 1
        self.acc_reward += Reward
        if self.current_step >= self.discrete_steps:
            terminated = True
        else:
            terminated = False
        if math.isnan(Reward):
            truncated = True
        else:
            truncated = False
        info = {"step ": self.current_step , " reward " : Reward}
        if MPC_purpose:
            #MW_comp = np.array([2, 28.01, 17.03])*1e-3
            print(y)
            return self.Tg, self.Tc, self.P_sep, self.Nl_sep, self.z_prod, self.Tbed_0, States["concentrations"], Reward, terminated
        else:
            return States, Reward, terminated, info

    def reset(self, seed = None, options = []):
        #super().reset(seed=seed)
        #global reward_hist
        self.current_step = 0
        self.Tg = np.copy(self.init_Tg)
        self.Tc = np.copy(self.init_Tc)
        self.sp_ratios = self.sp_ratios_init
        self.Fcyc = self.feed_flowrate*self.max_RR
        self.recycle_stream = np.zeros(6)
        x_init, Reward = self._env_step(_is_reset = True)
        #info = "reset complete"
        #print("accumulated reward", self.acc_reward)
        '''
        if self.acc_reward != 0:
            self.reward_hist = np.append(self.reward_hist, self.acc_reward)
            display.clear_output(wait=True)
            plt.plot(self.reward_hist)
            plt.plot(np.cumsum(self.reward_hist)/np.arange(1,self.reward_hist.size+1))
            plt.xlabel('Best: %g at epsiode %d' % (np.max(self.reward_hist),np.argmax(self.reward_hist)))
            display.display(plt.gcf())
        self.acc_reward = 0
        '''
        return x_init, Reward
        
    def get_reward_hist(self):
        
        return self.reward_hist
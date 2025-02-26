import numpy as np
# from pymoo.core.problem import Problem
from scipy.interpolate import griddata
import pandas as pd

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

class JPEGProblem(ElementwiseProblem):

    def __init__(self,snr,cmp,sample_points,cmp_samples,snr_samples):
        self.snr = snr
        self.cmp = cmp
        self.sample_points = sample_points
        self.cmp_samples = cmp_samples
        self.snr_samples = snr_samples
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=[0,50], xu=[35,100],vtype=int)
        # super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=[0,50], xu=[35,100])

    def interpolated_constraint_cmp(self,xy):
        return griddata(self.sample_points, self.cmp_samples, (float(xy[0])/100, xy[1]), method='linear')

    def interpolated_constraint_snr(self,xy):
        return griddata(self.sample_points, self.snr_samples, (float(xy[0])/100, xy[1]), method='linear')

    def _evaluate(self, x, out, *args, **kwargs):
        obj = -self.interpolated_constraint_cmp(x)

        constraint_1 = self.cmp - self.interpolated_constraint_cmp(x)
        constraint_2 = self.snr - self.interpolated_constraint_snr(x) 


        penalty1 = 1e6 * max(0, constraint_1)  # Large penalty for violation
        penalty2 = 1e6 * max(0, constraint_2)  # Large penalty for violation

        out["F"] = obj +penalty1+penalty2
        # if constraint_1 > 0 :  # If constraint is violated
        #     out["F"] += penalty * constraint_1 

        # if constraint_2 > 0:  # If constraint is violated
        #     out["F"] +=self.interpolated_constraint_snr(x) /40

        out["G"] = np.array([
            constraint_1,constraint_2
        ])


class Manager():

    def __init__(self):
        self.cmp_samples={}
        self.snr_samples={}
        self.test_pruning = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
        self.test_quality = [100, 90, 80, 70, 60, 50]
        self.test_points=[]
        self.window_size= 3
        self.test_counter = 0

        self.manager_cmp =-1
        self.manager_snr = -1
        for n in range(self.window_size):
            for p in self.test_pruning:
                for q in self.test_quality:
                    self.test_points.append((p,q))

        self.raw_tensor_size = 128*26*26*4*8 # in bits
        self.available_transmission_time = 0.010 # s
        self.solution_feasiable = 0
        
        # ToDo: update drop curve
        coef_map = [-2.15430309e-05,  2.53090430e-03, -1.10683795e-01,  2.17196770e+00, -1.94865597e+01,  1.09932162e+02]
        coef_sens = [-3.37021127e-05,  3.59472798e-03, -1.40599955e-01,  2.46219543e+00,-2.03229254e+01,  1.13382156e+02]
        self.curve_map = np.poly1d(coef_map)
        self.curve_sens = np.poly1d(coef_sens)
        
        # Algorithm configurations
        # self.algorithm = GA(pop_size=20)
        self.algorithm = GA(pop_size=20,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
            )
        self.termination = DefaultSingleObjectiveTermination(
                xtol=1e-8,
                cvtol=1e-6,
                ftol=1e-6,
                period=20,
                n_max_gen=20,
                n_max_evals=10000
            )
        
    def get_configuration(self):
        
        if self.test_counter < len(self.test_points):
            config = self.test_points[self.test_counter-1]
            return config[0], config[1]
        else:
            return  self.target_pruning,self.target_quality
        
    def get_compression_technique(self):
        return 1
    
    def get_pruning_threshold(self):
        return self.target_pruning
    
    def get_target_quality(self):
        return self.target_quality
    
    def get_intermedia_measurements(self):
        return self.manager_cmp,self.manager_snr
        # return self.target_cmp,self.target_snr
    
    def get_feasibility(self):
        return self.solution_feasiable
        
    def update_requirements(self,tolerable_mAP_drop, available_bandwidth): # [%, bps]
        available_bandwidth = available_bandwidth*0.5
        self.target_cmp = self.raw_tensor_size / (available_bandwidth*self.available_transmission_time)
        self.target_snr = self.get_snr_from_mapDrop(tolerable_mAP_drop)

        self.test_counter+=1
        # Define optimization problem
        if self.test_counter >= len(self.test_points):
            s_points = list(self.snr_samples.keys())
            s_snrs = np.mean(np.array(list(self.snr_samples.values())),axis=1)
            s_cmps = np.min(np.array(list(self.cmp_samples.values())),axis=1)
            print("Target snr, cmp:",self.target_snr,self.target_cmp)
            problem =JPEGProblem(self.target_snr,self.target_cmp,s_points,s_cmps,s_snrs)
            result = minimize(problem, self.algorithm, termination=self.termination,seed=1,verbose=False)
            print(result.G)
            print("Best solution found: \nX = %s\nF = %s" % (result.X, result.F))
            

            try:
                self.target_pruning = result.X[0]/100
                self.target_quality = result.X[1]
                self.solution_feasiable = 1
                self.manager_cmp= griddata(s_points, s_cmps, ( result.X[0]/100,  result.X[1]), method='linear')
                self.manager_snr= griddata(s_points, s_snrs, ( result.X[0]/100,  result.X[1]), method='linear')
            except:
                self.target_quality = 70
                self.target_pruning =0.1
                self.solution_feasiable = 0
        else:
            self.target_quality = -1
            self.target_pruning =-1
            self.solution_feasiable = -1
        # return self.target_quality, self.target_pruning

    def get_snr_from_mapDrop(self,mAP_drop):
        for snr in range(40):
            drop = self.curve_map([40-snr])
            if drop[0]>mAP_drop:
                return 41-snr
        return 0


    def update_sample_points(self, point, cmp, snr):
        try:
            snrs = self.snr_samples[point]
            snrs = np.roll(snrs,1)
            snrs[0] = snr
            self.snr_samples[point] = snrs

            cmps = self.cmp_samples[point]
            cmps = np.roll(cmps,1)
            cmps[0] = cmp
            self.cmp_samples[point] = cmps
        except:
            self.snr_samples[point] = np.ones(self.window_size)*snr
            self.cmp_samples[point] = np.ones(self.window_size)* cmp


        





    
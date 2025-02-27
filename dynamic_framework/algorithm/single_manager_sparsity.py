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
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=[0,60], xu=[99,100],vtype=int)

    def interpolated_constraint_cmp(self,xy):
        return griddata(self.sample_points, self.cmp_samples, (xy[0]/100, xy[1]), method='cubic')

    def interpolated_constraint_snr(self,xy):
        return griddata(self.sample_points, self.cmp_samples, (xy[0]/100, xy[1]), method='linear')

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[0]/100-x[1]/100

        constraint_1 = self.cmp-self.interpolated_constraint_cmp(x)
        constraint_2 = self.snr-self.interpolated_constraint_snr(x)

        out["G"] = np.array([
            constraint_1,constraint_2
        ])
        penalty = 1000  # Large penalty factor
        if constraint_1 > 0:  # If constraint is violated
            out["F"] += penalty * constraint_1
        if constraint_1 > 0:  # If constraint is violated
            out["F"] += penalty * constraint_2


class Manager():

    def __init__(self):
        self.raw_tensor_size = 128*26*26*4*8 # in bits
        self.available_transmission_time = 0.010 # s
        self.solution_feasiable = 0
        
        # SNR2mAP curve
        coef_map = [-2.15430309e-05,  2.53090430e-03, -1.10683795e-01,  2.17196770e+00, -1.94865597e+01,  1.09932162e+02]
        coef_sens = [-3.37021127e-05,  3.59472798e-03, -1.40599955e-01,  2.46219543e+00,-2.03229254e+01,  1.13382156e+02]
        self.curve_map = np.poly1d(coef_map)
        self.curve_sens = np.poly1d(coef_sens)

        # interpolation settings
        self.sample_space =     [[0.1, 60],   [0.1,100],    [0.99,60],    [0.99,100],   [0.9,60],   [0.9,100]]
        self.sample_space_snr = [ 5,        35,         0,                 10,          4,          20]
        self.sample_space_cmp = [ 48,       5,         80,                 30,          50,         5]
        self.history_window_size = 5
        self.history_counter =0
        self.history_point=[]
        self.history_snr=[]
        self.history_cmp=[]
        
        # Algorithm configurations
        self.algorithm = GA(pop_size=25,
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
        return self.target_quality, self.target_sparsity
    
    def get_intermedia_measurements(self):
        return self.target_cmp,self.target_snr
    
    def get_feasibility(self):
        return self.solution_feasiable
        
    def update_requirements(self,tolerable_mAP_drop, available_bandwidth): # [%, bps]
        available_bandwidth = available_bandwidth*0.6
        self.target_cmp = self.raw_tensor_size / (available_bandwidth*self.available_transmission_time)
        self.target_snr = self.get_snr_from_mapDrop(tolerable_mAP_drop)
        problem =JPEGProblem(self.target_snr,
                             self.target_cmp,
                             self.sample_space+self.history_point,
                             self.sample_space_cmp+self.history_cmp,
                             self.sample_space_snr+self.history_snr)
        result = minimize(problem, self.algorithm, termination=self.termination)

        try:
            self.target_sparsity= result.X[0]/100
            self.target_quality = result.X[1]
            self.solution_feasiable = 1
        except:
            self.target_sparsity =0.99
            self.target_quality = 60
            self.solution_feasiable = 0
        # return self.target_quality, self.target_pruning

    def get_snr_from_mapDrop(self,mAP_drop):
        for snr in range(40):
            drop = self.curve_map([40-snr])
            if drop[0]>mAP_drop:
                return 41-snr
        return 0


    def update_sample_points(self, point, cmp, snr):
        if len(self.history_point)<self.history_window_size:
            self.history_point.append(point)
            self.history_cmp.append(cmp)
            self.history_snr.append(snr)
        else:
            index = self.history_counter%self.history_window_size
            self.history_point[index] = point
            self.history_cmp[index] = cmp
            self.history_snr[index] = snr

        self.history_counter+=1





    
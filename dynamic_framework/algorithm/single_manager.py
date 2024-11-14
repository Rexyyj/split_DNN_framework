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

    def __init__(self,snr,cmp):
        self.snr = snr
        self.cmp = cmp
        self.sample_points, self.cmp_samples, self.snr_samples = self.get_jpeg_samples()
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=[60,5], xu=[100,35],vtype=int)

    def interpolated_constraint_cmp(self,xy):
        return griddata(self.sample_points, self.cmp_samples, (xy[0], xy[1]/100), method='cubic')

    def interpolated_constraint_snr(self,xy):
        return griddata(self.sample_points, self.cmp_samples, (xy[0], xy[1]/100), method='cubic')

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = x[1]/35-x[0]/100

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
        self.sample_points, self.cmp_samples, self.snr_samples = self.get_jpeg_samples()
        self.raw_tensor_size = 128*16*16*4*8 # in bits
        self.available_transmission_time = 0.010 # s
        
        # SNR2mAP curve
        coef_map = [-2.15430309e-05,  2.53090430e-03, -1.10683795e-01,  2.17196770e+00, -1.94865597e+01,  1.09932162e+02]
        coef_sens = [-3.37021127e-05,  3.59472798e-03, -1.40599955e-01,  2.46219543e+00,-2.03229254e+01,  1.13382156e+02]
        self.curve_map = np.poly1d(coef_map)
        self.curve_sens = np.poly1d(coef_sens)
        
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
                n_max_gen=10,
                n_max_evals=10000
            )
        
    def get_configuration(self,tolerable_mAP_drop, available_bandwidth): # [%, bps]
        self.target_cmp = self.raw_tensor_size / (available_bandwidth*self.available_transmission_time)
        self.target_snr = self.get_snr_from_mapDrop(tolerable_mAP_drop)

        problem =JPEGProblem(snr=self.target_snr,cmp=self.target_cmp)
        result = minimize(problem, self.algorithm, termination=self.termination)

        if result.X == None:
            self.target_quality = 60
            self.target_pruning =0.35
        else:
            self.target_quality = result.X[0]
            self.target_pruning = result.X[1]/100
        return self.target_quality, self.target_pruning

    def get_snr_from_mapDrop(self,mAP_drop):
        for snr in range(40):
            drop = self.curve_map([40-snr])
            if drop[0]>mAP_drop:
                return 41-snr
        return 0

        
    def get_jpeg_samples(self):
        jpeg_cha = pd.read_csv("/home/rex/gitRepo/split_DNN_framework/stable_tests/measurements/jpeg_snr_cha/characteristic.csv")
        jpeg_cha = jpeg_cha[jpeg_cha["sparsity"]>0]
        jpeg_cha = jpeg_cha[jpeg_cha["datasize_est"]>0]
        tensor_size = 128*26*26 *4
        jpeg_cha["ratio"] = tensor_size/jpeg_cha["datasize_est"]
        pruning = [0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35]
        quality = [60,70,80,90,100]

        sample_points=[]
        cmp_sample_values = []
        snr_sample_values = []

        cha_df_group =jpeg_cha.groupby("pruning_thresh")
        for p in pruning:
            cha_df = cha_df_group.get_group(p)
            cha_quality_df = cha_df.groupby("quality")
            for q in quality:
                cha_plot_df= cha_quality_df.get_group(q)
                sample_points.append([q,p])
                cmp_sample_values.append(cha_plot_df["ratio"].mean())
                snr_sample_values.append(cha_plot_df["reconstruct_snr"].mean())

        return np.array(sample_points), np.array(cmp_sample_values),np.array(snr_sample_values)




    
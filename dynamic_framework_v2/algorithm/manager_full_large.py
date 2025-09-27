import numpy as np
# from pymoo.core.problem import Problem
from scipy.interpolate import griddata
import pandas as pd
import random

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

class Problem(ElementwiseProblem):

    def __init__(self,snr,cmp,sample_points,cmp_samples,snr_samples, lb, ub):
        self.snr = snr
        self.cmp = cmp
        self.sample_points = sample_points
        self.cmp_samples = cmp_samples
        self.snr_samples = snr_samples
        super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=lb, xu=ub,vtype=int)
        # super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=[0,50], xu=[35,100],vtype=int)
        # super().__init__(n_var=2, n_obj=1, n_ieq_constr=2, xl=[0,50], xu=[35,100])

    def interpolated_cmp(self,xy):
        return griddata(self.sample_points, self.cmp_samples, (float(xy[0])/100, xy[1]), method='linear')

    def interpolated_snr(self,xy):
        return griddata(self.sample_points, self.snr_samples, (float(xy[0])/100, xy[1]), method='linear')

    def _evaluate(self, x, out, *args, **kwargs):
        obj = -self.interpolated_cmp(x)

        constraint_1 = self.cmp - self.interpolated_cmp(x)
        constraint_2 = self.snr - self.interpolated_snr(x) 

        penalty1 = 1e6 * max(0, constraint_1)  # Large penalty for violation
        penalty2 = 1e6 * max(0, constraint_2)  # Large penalty for violation

        out["F"] = obj +penalty1+penalty2

        out["G"] = np.array([
            constraint_1,constraint_2
        ])


class Manager():

    def __init__(self):
        self.jpeg_cmp_samples={}
        self.jpeg_snr_samples={}
        self.decom_cmp_samples={}
        self.decom_snr_samples={}
        self.reg_cmp_samples={}
        self.reg_snr_samples={}

        self.test_pruning = [0, 0.05, 0.1, 0.15, 0.2,0.25]
        self.jpeg_quality = [100, 90, 80, 70, 60]
        # self.regression_quality = [1,2,3,4,5]
        # self.decomposition_quality = [1,2,3,4,5]

        self.test_points=[]
        self.window_size= 2
        self.test_counter = 0

        self.manager_cmp =-1
        self.manager_snr = -1
        self.target_cmp =-1
        self.target_snr = -1
        self.target_technique = -1
        self.target_transmission_time = -1

        self.result_jpeg= None
        self.result_decom = None
        self.result_reg = None

        for n in range(self.window_size):
            for p in self.test_pruning:
                for q_j in self.jpeg_quality:
                    self.test_points.append((1,p,q_j))
                for q_j in self.jpeg_quality:
                    self.test_points.append((1,p,q_j))
                for q_j in self.jpeg_quality:
                    self.test_points.append((1,p,q_j))
                # if p <=0.2:
                # for q_d in self.decomposition_quality:
                #     self.test_points.append((2, p,q_d))
                # for q_r in self.regression_quality:
                #     self.test_points.append((3,p,q_r))
        random.seed(317462)
        random.shuffle(self.test_points)

        self.raw_tensor_size = 128*26*26*4*8 # in bits
        self.solution_feasiable = 0
        
        # ToDo: update drop curve

        # STMARC [k, h, b]
        # self.map_curve = [0.059, 0.546, 2]
        # self.sen_curve = [0.064, 0.622, 2]

        # BEV [k, h, b]
        # self.map_curve = [0.218, 1.674, 0]
        # self.sen_curve = [0.182, 1.044, 0]
        
        # vidVRD [k, h, b]
        self.map_curve = [0.176, 2.385, 6]
        self.sen_curve = [0.210, 3.451, 6]

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
        
    def set_compression_technique(self,tech):
        self.target_technique = tech
    
    def set_pruning_threshold(self, thresh):
        self.target_pruning =thresh
    
    def set_compression_quality(self, qua):
        self.target_quality= qua
        
    def get_compression_technique(self):
        return self.target_technique
    
    def get_pruning_threshold(self):
        return self.target_pruning
    
    def get_compression_quality(self):
        return self.target_quality
    
    def get_target_cmp(self):
        return self.target_cmp
    
    def get_target_snr(self):
        return self.target_snr
    
    def get_est_cmp(self):
        return self.manager_cmp

    def get_est_snr(self):
        return self.manager_snr
    
    def get_intermedia_measurements(self):
        return self.manager_cmp,self.manager_snr
        # return self.target_cmp,self.target_snr
    
    def get_feasibility(self):
        return self.solution_feasiable
    
    def get_testing_frame_length(self):
        return len(self.test_points)
    
    def get_result_jpeg_f(self):
        try:
            return str(self.result_jpeg.X[0]/100)+"/"+str(self.result_jpeg.X[1])+"/"+str(-self.result_jpeg.F[0])
        except:
            return "-1/-1/-1"
    
    def get_result_decom_f(self):
        try:
            return str(self.result_decom.X[0]/100)+"/"+str(self.result_decom.X[1])+"/"+str(-self.result_decom.F[0] )
        except:
            return "-1/-1/-1"
    
    def get_result_reg_f(self):
        try:
            return str(self.result_reg.X[0]/100)+"/"+str(self.result_reg.X[1])+"/"+str(-self.result_reg.F[0] )
        except:
            return "-1/-1/-1"
        
    def get_transmission_time(self):
        return self.target_transmission_time
        
    def update_requirements(self,tolerable_mAP_drop, target_fps,available_bandwidth, f_index): # [%, bps]
        # available_bandwidth = available_bandwidth*0.5
        # self.target_cmp = self.raw_tensor_size / (available_bandwidth*self.available_transmission_time)
        # self.target_snr = self.get_snr_from_mapDrop(tolerable_mAP_drop,tolerable_mAP_drop) # use same drop for mAP and sen

        # self.test_counter+=1
        # Define optimization problem
        if f_index >= len(self.test_points):
            
            jpeg_transmission_time = 1/target_fps - 0.016 - 0.01
            decom_transmission_time = 1/target_fps - 0.016 - 0.1
            reg_transmission_time = 1/target_fps - 0.016 - 0.1
            # jpeg optimization
            if jpeg_transmission_time>0:
                jpeg_target_cmp = self.raw_tensor_size / (available_bandwidth*jpeg_transmission_time)
                jpeg_target_snr = self.get_snr_from_mapDrop(tolerable_mAP_drop,tolerable_mAP_drop) # use same drop for mAP 
                s_points_jpeg = list(self.jpeg_snr_samples.keys())
                s_snrs_jpeg = np.mean(np.array(list(self.jpeg_snr_samples.values())),axis=1)
                s_cmps_jpeg = np.min(np.array(list(self.jpeg_cmp_samples.values())),axis=1)
                problem_jpeg =Problem(jpeg_target_snr,jpeg_target_cmp,s_points_jpeg,s_cmps_jpeg,s_snrs_jpeg,[0,60],[35, 100])
                self.result_jpeg = minimize(problem_jpeg, self.algorithm, termination=self.termination,seed=1,verbose=False)
                try:
                    jpeg_manager_cmp= griddata(s_points_jpeg, s_cmps_jpeg, ( self.result_jpeg.X[0]/100,  self.result_jpeg.X[1]), method='linear')
                    jpeg_manager_snr= griddata(s_points_jpeg, s_snrs_jpeg, ( self.result_jpeg.X[0]/100,  self.result_jpeg.X[1]), method='linear')
                except:
                    self.result_jpeg = None
                    jpeg_manager_cmp=-1
                    jpeg_manager_snr=-1
            else:
                self.result_jpeg = None
                jpeg_manager_cmp=-1
                jpeg_manager_snr=-1
            # decomposition optimization
            if decom_transmission_time>0:
                decom_target_cmp =  self.raw_tensor_size/ (available_bandwidth*decom_transmission_time)
                decom_target_snr = self.get_snr_from_mapDrop(tolerable_mAP_drop,tolerable_mAP_drop) # use same drop 
                s_points_decom = list(self.decom_snr_samples.keys())
                s_snrs_decom = np.mean(np.array(list(self.decom_snr_samples.values())),axis=1)
                s_cmps_decom = np.min(np.array(list(self.decom_cmp_samples.values())),axis=1)
                problem_decom =Problem(decom_target_snr,decom_target_cmp,s_points_decom,s_cmps_decom,s_snrs_decom,[0,1],[25, 5])
                self.result_decom = minimize(problem_decom, self.algorithm, termination=self.termination,seed=1,verbose=False)
                try:
                    decom_manager_cmp= griddata(s_points_decom, s_cmps_decom, ( self.result_decom.X[0]/100,  self.result_decom.X[1]), method='linear')
                    decom_manager_snr= griddata(s_points_decom, s_snrs_decom, ( self.result_decom.X[0]/100,  self.result_decom.X[1]), method='linear')
                except:
                    self.result_decom= None
                    decom_manager_cmp=-1
                    decom_manager_snr=-1
            else:
                self.result_decom= None
                decom_manager_cmp=-1
                decom_manager_snr=-1
            # Regression optimization
            if reg_transmission_time>0:
                reg_target_cmp =  self.raw_tensor_size/ (available_bandwidth*reg_transmission_time)
                reg_target_snr = self.get_snr_from_mapDrop(tolerable_mAP_drop*0.8,tolerable_mAP_drop*0.8) # use same drop 
                s_points_reg = list(self.reg_snr_samples.keys())
                s_snrs_reg = np.mean(np.array(list(self.reg_snr_samples.values())),axis=1)
                s_cmps_reg = np.min(np.array(list(self.reg_cmp_samples.values())),axis=1)
                problem_reg =Problem(reg_target_snr,reg_target_cmp,s_points_reg,s_cmps_reg,s_snrs_reg,[0,1],[35, 6])
                self.result_reg = minimize(problem_reg, self.algorithm, termination=self.termination,seed=1,verbose=False)
                try:
                    reg_manager_cmp= griddata(s_points_reg, s_cmps_reg, ( self.result_reg.X[0]/100,  self.result_reg.X[1]), method='linear')
                    reg_manager_snr= griddata(s_points_reg, s_snrs_reg, ( self.result_reg.X[0]/100,  self.result_reg.X[1]), method='linear')
                except:
                    self.result_reg = None
                    reg_manager_cmp=-1
                    reg_manager_snr=-1
            else:
                self.result_reg = None
                reg_manager_cmp=-1
                reg_manager_snr=-1
            

            result = None
            if self.result_jpeg!=None or self.result_decom!=None or self.result_reg!=None:
                self.solution_feasiable = 1
                if self.result_jpeg != None:
                    result = self.result_jpeg
                    self.target_cmp = jpeg_target_cmp
                    self.target_snr = jpeg_target_snr
                    self.target_transmission_time=jpeg_transmission_time
                    self.manager_cmp = jpeg_manager_cmp
                    self.manager_snr =  jpeg_manager_snr
                    self.target_technique = 1
                
                if self.result_decom!= None:
                    if result == None:
                        result = self.result_decom
                        self.target_cmp = decom_target_cmp
                        self.target_snr = decom_target_snr
                        self.target_transmission_time = decom_transmission_time
                        self.manager_cmp = decom_manager_cmp
                        self.manager_snr =  decom_manager_snr
                        self.target_technique=2
                    elif self.result_decom.F < result.F:
                        result = self.result_decom
                        self.target_cmp = decom_target_cmp
                        self.target_snr = decom_target_snr
                        self.target_transmission_time =decom_transmission_time
                        self.manager_cmp = decom_manager_cmp
                        self.manager_snr =  decom_manager_snr
                        self.target_technique=2

                if self.result_reg!=None:
                    if result == None:
                        result = self.result_reg
                        self.target_cmp = reg_target_cmp
                        self.target_snr = reg_target_snr
                        self.target_transmission_time = reg_transmission_time
                        self.manager_cmp = reg_manager_cmp
                        self.manager_snr =  reg_manager_snr
                        self.target_technique= 3
                    elif self.result_reg.F< result.F:
                        result = self.result_reg
                        self.target_cmp = reg_target_cmp
                        self.target_snr = reg_target_snr
                        self.target_transmission_time=reg_transmission_time
                        self.manager_cmp = reg_manager_cmp
                        self.manager_snr =  reg_manager_snr
                        self.target_technique=3

                self.target_pruning = result.X[0]/100
                self.target_quality = result.X[1]
            else:
                self.solution_feasiable = 0

        else:
            config = self.test_points[f_index-1]
            self.target_technique= config[0]
            self.target_pruning =config[1]
            self.target_quality = config[2]
            self.target_transmission_time=-1
            self.solution_feasiable = -1
        # return self.target_quality, self.target_pruning

    def get_snr_from_mapDrop(self,mAP_drop, sen_drop):
        map_snr = 0
        sen_snr = 0
        for snr in range(50):
            drop = self.get_drop_from_snr(50-snr, self.map_curve[0],self.map_curve[1],self.map_curve[2])
            if drop>mAP_drop:
                map_snr = 51-snr
                break

        for snr in range(50):
            drop = self.get_drop_from_snr(50-snr, self.sen_curve[0],self.sen_curve[1],self.sen_curve[2])
            if drop>sen_drop:
                sen_snr = 51-snr
                break
        
        return max(map_snr, sen_snr)

    def get_drop_from_snr(self, snr, k, h, b):
        if snr<b:
            return 1
        else:
            return h* (np.e **(-k*snr))

    def update_sample_points(self, tech, point, cmp, snr):
        if tech ==1:## insert jpeg points
            try:
                snrs = self.jpeg_snr_samples[point]
                snrs = np.roll(snrs,1)
                snrs[0] = snr
                self.jpeg_snr_samples[point] = snrs

                cmps = self.jpeg_cmp_samples[point]
                cmps = np.roll(cmps,1)
                cmps[0] = cmp
                self.jpeg_cmp_samples[point] = cmps
            except:
                self.jpeg_snr_samples[point] = np.ones(self.window_size)*snr
                self.jpeg_cmp_samples[point] = np.ones(self.window_size)* cmp
        elif tech ==2: ## inset decomposition points
            try:
                snrs = self.decom_snr_samples[point]
                snrs = np.roll(snrs,1)
                snrs[0] = snr
                self.decom_snr_samples[point] = snrs

                cmps = self.decom_cmp_samples[point]
                cmps = np.roll(cmps,1)
                cmps[0] = cmp
                self.decom_cmp_samples[point] = cmps
            except:
                self.decom_snr_samples[point] = np.ones(self.window_size)*snr
                self.decom_cmp_samples[point] = np.ones(self.window_size)* cmp
        elif tech ==3: ## insert regression points
            try:
                snrs = self.reg_snr_samples[point]
                snrs = np.roll(snrs,1)
                snrs[0] = snr
                self.reg_snr_samples[point] = snrs

                cmps = self.reg_cmp_samples[point]
                cmps = np.roll(cmps,1)
                cmps[0] = cmp
                self.reg_cmp_samples[point] = cmps
            except:
                self.reg_snr_samples[point] = np.ones(self.window_size)*snr
                self.reg_cmp_samples[point] = np.ones(self.window_size)* cmp
        else:
            raise Exception("Unknow sample points")

        





    
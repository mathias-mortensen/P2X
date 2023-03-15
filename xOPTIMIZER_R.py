#PACKAGES

import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd 
import numpy as np
import os 
import sys
from pathlib import Path
print(os.getcwd())

#---------------------------------------------------------------------------
from xFunctions import *  # Contains all custom function created for the model
#---------------------------------------------------------------------------
excel_path = sys.argv[1]
parent_folder = os.path.dirname(excel_path)
data_folder = parent_folder + "/Data"

# Input Data (from master excel file)
df_run = pd.read_excel(excel_path,'Run')
df_param = pd.read_excel(excel_path,'parameter settings')
df_pw = pd.read_excel(excel_path,'Efficiency breakpoints')


# Assigning model input values to variables
Start_date = assign_var(df_run,'Start_date_sim') 
End_date = assign_var(df_run,'End_date_sim')
Demand_pattern = assign_var(df_run,'Demand_pattern')
Start_date_scen = assign_var(df_run,'Start_date_scen')
End_date_scen = assign_var(df_run,'End_date_scen')
sEfficiency = assign_var(df_run,'sEfficiency')
n_samples = int(assign_var(df_run,'n_samples'))
blocksize = int(assign_var(df_run,'blocksize'))
n_clusters = int(assign_var(df_run,'n_clusters'))
PV_Cluster = assign_var(df_run,'PV_Cluster')
n_clusters_PV = int(assign_var(df_run,'n_clusters_PV'))
blocksize_PV = int(assign_var(df_run,'blocksize_PV'))
weeks = int(assign_var(df_run,'weeks'))

# constants assignment from 'Parameter settings'
hourly_demand = assign_var(df_param,'k_d')



#Importing solar data
P_PV_max = import_PV(data_folder+'/PV_data.xlsx',Start_date, End_date, Start_date_scen, End_date_scen)
# importing day ahead prices as well as the applicaple time range detemrined by the date settings
DA,TimeRange = import_DA(data_folder+'/Elspotprices_RAW.csv',Start_date, End_date, Start_date_scen, End_date_scen)

Demand = demand_assignment(Demand_pattern,TimeRange,hourly_demand)

# importing FCR data
file_FCR = assign_var(df_run,'FCR price file name')
price_column_FCR = assign_var(df_run,'FCR price column')
time_column_FCR = assign_var(df_run,'FCR time column')
c_FCR = import_generic(file_FCR,data_folder,price_column_FCR, time_column_FCR, Start_date, End_date, Start_date_scen, End_date_scen)
# importing mFRR data
c_mFRR = import_mFRR(data_folder+"/MfrrReservesDK1.csv", Start_date, End_date, Start_date_scen, End_date_scen)
file_mFRR = assign_var(df_run,'mFRR price file name')
price_column_mFRR = assign_var(df_run,'mFRR price column')
time_column_mFRR = assign_var(df_run,'mFRR time column')
c_mFRR_2 = import_generic(file_mFRR,data_folder,price_column_mFRR, time_column_mFRR, Start_date, End_date, Start_date_scen, End_date_scen)
# importing aFRR data using the generic function
file_aFRR_up = assign_var(df_run,'aFRR_up price file name')
price_column_aFRR_up = assign_var(df_run,'aFRR_up price column')
time_column_aFRR_up = assign_var(df_run,'aFRR_up time column')
c_aFRR_upX = import_generic(file_aFRR_up,data_folder,price_column_aFRR_up, time_column_aFRR_up, Start_date, End_date, Start_date_scen, End_date_scen)
file_aFRR_down = assign_var(df_run,'aFRR_down price file name')
price_column_aFRR_down = assign_var(df_run,'aFRR_down price column')
time_column_aFRR_down = assign_var(df_run,'aFRR_down time column')
c_aFRR_down = import_generic(file_aFRR_down,data_folder,price_column_aFRR_down, time_column_aFRR_down, Start_date, End_date, Start_date_scen, End_date_scen)

#PV data import
file_PV = assign_var(df_run,'Solar irradiance')
power_column_PV = assign_var(df_run,'PV power column')
time_column_PV = assign_var(df_run,'PV time column')
P_PV_max = import_generic(file_PV,data_folder,power_column_PV, time_column_PV, Start_date, End_date, Start_date_scen, End_date_scen)


# Converting the efficiency breakpoints to respective lists for setpoints and resulting hydrogen mass flow
if sEfficiency == 'pw':
    pem_setpoint = df_pw['p_pem'].tolist() 
    hydrogen_mass_flow = df_pw['m'].tolist()



print("Hello")



"""
# MODEL 1 Included for testing purposes ------------------------------------------------

solver = po.SolverFactory('gurobi')
model = pe.ConcreteModel()

#set t in T
T = len(P_PV_max)
model.T = pe.RangeSet(1,T)


#initializing parameters
model.P_PV_max = pe.Param(model.T, initialize=P_PV_max)
model.DA = pe.Param(model.T, initialize=DA)
model.m_demand = pe.Param(model.T, initialize = Demand)

model.P_pem_cap = assign_var(df_param,'P_pem_cap')
model.P_pem_min = assign_var(df_param,'P_pem_min')
model.P_com = assign_var(df_param,'P_com')
model.P_grid_cap = assign_var(df_param,'P_grid_cap')
model.eff = assign_var(df_param,'eff')
model.r_in = assign_var(df_param,'r_in')
model.r_out = assign_var(df_param,'r_out')
model.k_d = k_d
model.S_Pu_max = S_Pu_max
model.S_raw_max = S_raw_max
model.m_H2_max = m_H2_max
model.ramp_pem = ramp_pem
model.ramp_com = ramp_com
model.P_PV_cap = P_PV_cap
model.PT = PT
model.CT = CT

#defining variables
model.p_grid = pe.Var(model.T, domain=pe.Reals)
model.p_PV = pe.Var(model.T, domain=pe.NonNegativeReals)
model.p_pem = pe.Var(model.T, domain=pe.NonNegativeReals, bounds=(0,52.5))
model.m_H2 = pe.Var(model.T, domain=pe.NonNegativeReals, bounds=(0,1100))
model.m_CO2 = pe.Var(model.T, domain=pe.NonNegativeReals)
model.m_H2O = pe.Var(model.T, domain=pe.NonNegativeReals)
model.m_Ri = pe.Var(model.T, domain=pe.NonNegativeReals)
model.m_Ro = pe.Var(model.T, domain=pe.NonNegativeReals)
model.m_Pu = pe.Var(model.T, domain=pe.NonNegativeReals)
model.s_raw = pe.Var(model.T, domain=pe.NonNegativeReals)
model.s_Pu = pe.Var(model.T, domain=pe.NonNegativeReals)
model.zT = pe.Var(model.T, domain = pe.Binary) #binary decision variable
model.cT = pe.Var(model.T, domain = pe.Reals)
model.c_obj = pe.Var(model.T, domain = pe.Reals)

#model.η = pe.Var(model.T, domain = pe.NonNegativeReals)
#Objective
expr = sum((model.DA[t]+model.cT[t])*model.p_grid[t] for t in model.T)
model.objective = pe.Objective(sense = pe.minimize, expr=expr)

#creating a set of constraints
model.c1 = pe.ConstraintList()
for t in model.T:
    model.c1.add(model.p_grid[t] + model.p_PV[t] == model.p_pem[t] + model.P_com)

#Constraint 2.1
model.c2_1 = pe.ConstraintList()
for t in model.T:
    model.c2_1.add(-model.P_grid_cap <= model.p_grid[t])

#Constraint 2.2
model.c2_2 = pe.ConstraintList()
for t in model.T:
    model.c2_2.add(model.p_grid[t] <= model.P_grid_cap)

#Constraint 3
model.c3_1 = pe.ConstraintList()
for t in model.T:
    model.c3_1.add(0 <= model.p_PV[t])

#Constraint 3
model.c3_2 = pe.ConstraintList()
for t in model.T:
    model.c3_2.add(model.p_PV[t] <= model.P_PV_max[t])

model.c4_1 = pe.ConstraintList()
for t in model.T:
    model.c4_1.add(model.P_pem_min <= model.p_pem[t])

model.c4_2 = pe.ConstraintList()
for t in model.T:
    model.c4_2.add(model.p_pem[t] <= model.P_pem_cap)


if sEfficiency == 'pw':
  model.c_piecewise = Piecewise(  model.T,
                          model.m_H2,model.p_pem,
                        pw_pts=pem_setpoint,
                        pw_constr_type='EQ',
                        f_rule=hydrogen_mass_flow,
                        pw_repn='SOS2')
                   
if sEfficiency == 'k':
  model.csimple = pe.ConstraintList()
  for t in model.T:
      model.csimple.add(model.p_pem[t] == model.m_H2[t]/model.eff)



model.c6 = pe.ConstraintList()
for t in model.T:
    model.c6.add(model.m_CO2[t] == model.r_in*model.m_H2[t])

model.c7 = pe.ConstraintList()
for t in model.T:
    model.c7.add(model.m_Ri[t] == model.m_H2[t] + model.m_CO2[t])

model.c8 = pe.ConstraintList()
for t in model.T:
    model.c8.add(model.m_Ro[t] == model.m_Pu[t] + model.m_H2O[t])

model.c9 = pe.ConstraintList()
for t in model.T:
    model.c9.add(model.m_Pu[t] == model.r_out * model.m_H2O[t])

model.c10 = pe.ConstraintList()
for t in model.T:
    model.c10.add(model.m_Pu[t] == model.k_d)

#model.c11_1 = pe.ConstraintList()
#for t in model.T:
#    model.c11_1.add(0 <= model.s_raw[t])

model.c11_2 = pe.ConstraintList()
for t in model.T:
    model.c11_2.add(model.s_raw[t] <= model.S_raw_max)

model.c12 = pe.ConstraintList()
for t in model.T:
    if t >= 2:
        model.c12.add(model.s_raw[t] == model.s_raw[t-1] + model.m_Ri[t] - model.m_Ro[t])

model.c13_1 = pe.Constraint(expr=model.s_raw[1] == 0.5*model.S_raw_max + model.m_Ri[1] - model.m_Ro[1])

model.c13_2 = pe.Constraint(expr=0.5*model.S_raw_max == model.s_raw[T])

#model.c14_1 = pe.ConstraintList()
#for t in model.T:
#  model.c14_1.add(0 <= model.s_Pu[t])

model.c14_2 = pe.ConstraintList()
for t in model.T:
  model.c14_2.add(model.s_Pu[t] <= model.S_Pu_max)

model.c15 = pe.Constraint(expr = model.s_Pu[1] == model.m_Pu[1])

model.c16 = pe.ConstraintList()
for t in model.T:
  if t >= 2:
    model.c16.add(model.s_Pu[t] == model.s_Pu[t-1] + model.m_Pu[t] - model.m_demand[t])

model.c17_1 = pe.ConstraintList()
for t in model.T:
  if t >= 2:
    model.c17_1.add(-model.ramp_pem * model.P_pem_cap <= model.p_pem[t] - model.p_pem[t-1])

model.c17_2 = pe.ConstraintList()
for t in model.T:
  if t >= 2:
    model.c17_2.add(model.p_pem[t] - model.p_pem[t-1] <= model.ramp_pem * model.P_pem_cap)


model.c25_1 = pe.ConstraintList()
for t in model.T:
  model.c25_1.add(model.zT[t] >= -model.p_grid[t]/model.P_grid_cap)

model.c25_2 = pe.ConstraintList()
for t in model.T:
  model.c25_2.add(model.zT[t] <= 1-model.p_grid[t]/model.P_grid_cap)

model.c25_3 = pe.ConstraintList()
for t in model.T:
  model.c25_3.add(model.cT[t] == (1-model.zT[t])*model.CT - model.zT[t]*model.PT)

model.cObj = pe.ConstraintList()
for t in model.T:
  model.cObj.add(model.c_obj[t] == (model.DA[t]+model.cT[t])*model.p_grid[t])





#model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
results = solver.solve(model)
print(results)


# MODEL 1 Included for testing purposes ------------------------------------------------
"""
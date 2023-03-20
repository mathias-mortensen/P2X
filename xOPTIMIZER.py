#PACKAGES
import pyomo.environ as pe
import pyomo.opt as po
from pyomo.core import *
import pandas as pd
import numpy as np
#---------------------------------------------------------------------------
from xFunctions import *  # Contains all custom functions created for the model
#---------------------------------------------------------------------------

#---------------------------- Input Data (from master excel file) -------------------------------------

df_run = pd.read_excel('Opt_X.xlsx','Run')
df_param = pd.read_excel('Opt_X.xlsx','parameter settings')
df_pw = pd.read_excel('Opt_X.xlsx','Efficiency breakpoints')

# Assigning model input values to variables
version = assign_var(df_run,'model_version')
Start_date = assign_var(df_run,'Start_date_sim') 
End_date = assign_var(df_run,'End_date_sim')
Demand_pattern = assign_var(df_run,'Demand_pattern')
Start_date_scen = assign_var(df_run,'Start_date_scen')
End_date_scen = assign_var(df_run,'End_date_scen')
sEfficiency = assign_var(df_run,'sEfficiency')
n_samples = int(assign_var(df_run,'n_samples'))
blocksize = int(assign_var(df_run,'block size'))
PV_Cluster = assign_var(df_run,'PV_Cluster')
n_clusters_PV = int(assign_var(df_run,'n_clusters_PV'))
blocksize_PV = int(assign_var(df_run,'blocksize_PV'))
weeks = int(assign_var(df_run,'weeks'))

# constants assignment from 'Parameter settings'
hourly_demand = assign_var(df_param,'k_d')

#PV data import
file_PV = assign_var(df_run,'Solar irradiance')
power_column_PV = assign_var(df_run,'PV power column')
time_column_PV = assign_var(df_run,'PV time column')
P_PV_max = import_generic(file_PV,power_column_PV, time_column_PV, Start_date, End_date,'dict')

# importing day ahead prices as well as the applicaple time range detemrined by the date settings
# TIMERANGE MAY NOT BE NEEDED, 'DA' is obsolete, use c_DA
DA,TimeRange = import_DA('Elspotprices_RAW.csv',Start_date, End_date, Start_date_scen, End_date_scen)
file_DA = assign_var(df_run, 'Day-ahead file name')
price_column_DA = assign_var(df_run,'DA price column')
time_column_DA = assign_var(df_run,'DA time column')
c_DA = import_generic(file_DA,price_column_DA, time_column_DA, Start_date, End_date,'dict')

#Generating demand profile
Demand = demand_assignment(Demand_pattern,TimeRange,hourly_demand)

# importing FCR data
file_FCR = assign_var(df_run,'FCR price file name')
price_column_FCR = assign_var(df_run,'FCR price column')
time_column_FCR = assign_var(df_run,'FCR time column')
c_FCR = import_generic(file_FCR,price_column_FCR, time_column_FCR, Start_date, End_date,'dict')
# importing mFRR data
c_mFRR = import_mFRR("MfrrReservesDK1.csv", Start_date, End_date, Start_date_scen, End_date_scen)
file_mFRR = assign_var(df_run,'mFRR price file name')
price_column_mFRR = assign_var(df_run,'mFRR price column')
time_column_mFRR = assign_var(df_run,'mFRR time column')
c_mFRR = import_generic(file_mFRR,price_column_mFRR, time_column_mFRR, Start_date, End_date,'dict')
# importing aFRR data using the generic function
file_aFRR_up = assign_var(df_run,'aFRR_up price file name')
price_column_aFRR_up = assign_var(df_run,'aFRR_up price column')
time_column_aFRR_up = assign_var(df_run,'aFRR_up time column')
c_aFRR_up = import_generic(file_aFRR_up,price_column_aFRR_up, time_column_aFRR_up, Start_date, End_date,'dict')
file_aFRR_down = assign_var(df_run,'aFRR_down price file name')
price_column_aFRR_down = assign_var(df_run,'aFRR_down price column')
time_column_aFRR_down = assign_var(df_run,'aFRR_down time column')
c_aFRR_down = import_generic(file_aFRR_down,price_column_aFRR_down, time_column_aFRR_down, Start_date, End_date,'dict')

#------------------------------------- generating scenarios for the stochastic model -----------------------------------------
if version == 3:
    c_DA_scen = import_generic(file_DA,price_column_DA, time_column_DA, Start_date_scen, End_date_scen,'list')
    c_FCR_scen = import_generic(file_FCR,price_column_FCR, time_column_FCR, Start_date_scen, End_date_scen,'list')
    c_aFRR_up_scen = import_generic(file_aFRR_up,price_column_aFRR_up, time_column_aFRR_up, Start_date_scen, End_date_scen,'list')
    c_aFRR_down_scen = import_generic(file_aFRR_down,price_column_aFRR_down, time_column_aFRR_down, Start_date_scen, End_date_scen,'list')
    c_mFRR_scen = import_generic(file_mFRR,price_column_mFRR, time_column_mFRR, Start_date_scen, End_date_scen,'list')

    Data = [c_DA_scen, c_FCR_scen, c_aFRR_up_scen, c_aFRR_down_scen, c_mFRR_scen]
    Data_names = ['DA','FCR','aFRR Up','aFRR Down','mFRR']

    Data_comb = [c_DA_scen, c_aFRR_up_scen, c_aFRR_down_scen, c_mFRR_scen]
    Data_comb_names = ['DA','aFRR Up','aFRR Down','mFRR']

    sampling_method = assign_var(df_run,'scenario sampling method')
    blocksize = assign_var(df_run,'block size') #number of hours for each 
    sample_length = assign_var(df_run,'sample length') #number of hours for each scenario - should depend on time period ?
    n_clusters = int(assign_var(df_run,'number of clusters'))

    if sampling_method == 'single':
        scenarios = Bootsrap(sampling_method,Data,Data_names,n_samples,blocksize,sample_length)
        Rep_scen,Prob = K_Medoids(scenarios,n_clusters) #generating representative scenarios in "Rep_scen" from 'scenarios' with probabilities in "Prob"
        Φ, Ω,c_FCRs,c_aFRR_ups,c_aFRR_downs,c_mFRR_ups,c_DAs,π_r,π_DA = SingleInputData(Rep_scen,Prob)

    if sampling_method == 'combined':
        ## Generate Average Price for all markets for each time (Only for "Combined scenario generation"!!) ##
        ## For DA, aFRR_up & down and mFRR 
        scenarios = Bootsrap('combined',Data_comb,Data_comb_names,n_samples,blocksize,sample_length)
        Rep_scen_combALL,Prob_comb_all = scenario_comb(scenarios,n_samples,sample_length,n_clusters,c_FCR_scen,blocksize)
        Φ, Ω,c_FCRs,c_aFRR_ups,c_aFRR_downs,c_mFRR_ups,c_DAs,π_r,π_DA = SingleInputData(Rep_scen_combALL,Prob_comb_all)

# if running a deterministic model, "ONE" scenario is applied and the input data is written to scenario
elif version == 1 or version == 2:
    Φ = 1
    Ω = 1
    c_DAs = write_1d_to_2d(c_DA)
    π_DA = {}
    π_DA[1] = 1
    π_r = {}
    π_r[1] = 1
    if version == 2:
        c_FCRs =  write_1d_to_2d(c_FCR)
        c_aFRR_ups = write_1d_to_2d(c_aFRR_up)
        c_aFRR_downs = write_1d_to_2d(c_aFRR_down)
        c_mFRR_ups = write_1d_to_2d(c_mFRR)
        
    
#--------------------------------------- electrolyzer efficiency ----------------------------------------
# Converting the efficiency breakpoints to respective lists for setpoints and resulting hydrogen mass flow
# assigning max flow used to bound variable in model
if sEfficiency == 'pw':
    pem_setpoint = df_pw['p_pem'].tolist() 
    hydrogen_mass_flow = df_pw['m'].tolist()
    m_H2_max = max(hydrogen_mass_flow)
elif sEfficiency == 'k':
    m_H2_max = assign_var(df_param,'P_pem_cap') * assign_var(df_param,'eff')


# --------------------- choosing solver and  initializing model ------------------------------
solver = po.SolverFactory('gurobi')
model = pe.ConcreteModel()

# ------------------------ Defining the sets of the model ------------------------------------

T = len(P_PV_max) # hours in simulation period
model.T = pe.RangeSet(1,T)
model.Ω = pe.RangeSet(1,Ω)
model.Φ = pe.RangeSet(1,Φ)
if version == 2 or version == 3: # if including the reserve market in the model
    model.T_block = pe.RangeSet(1,T,4) #used to implement block length of FCR data


# --------------------------- Defining/initializing model parameters -----------------------------

model.P_PV_max = pe.Param(model.T, initialize=P_PV_max) # not scenario dependent
model.m_demand = pe.Param(model.T, initialize = Demand) # not scenario dependent
model.c_DA = pe.Param(model.Φ, model.T, initialize=c_DAs) # implement Φ = 1 for models 1 and 2, change DA to c_DA in models: model.DA = pe.Param(model.T, initialize=DA)
# c_DA = scenarios of version 3, else single input series

model.P_pem_cap = assign_var(df_param,'P_pem_cap') 
model.P_pem_min = assign_var(df_param,'P_pem_min')
model.P_com = assign_var(df_param,'P_com')
model.P_grid_cap = assign_var(df_param,'P_grid_cap')
model.eff = assign_var(df_param,'eff') # only needed if 'k'
model.r_in = assign_var(df_param,'r_in')

model.m_Pu = assign_var(df_param,'k_d') 
model.m_Ro = model.m_Pu * (1 + 18.01528/32.04) # The mass flow out of the storage is the pure methanol + water determined by a 1:1 molecular ratio and molar masses of methanol and water (in brackets) 
model.S_Pu_max = assign_var(df_param,'S_Pu_max') 
model.S_raw_max = assign_var(df_param,'S_raw_max')
model.m_H2_max = assign_var(df_param,'m_H2_max') # Not needed?
model.ramp_pem = assign_var(df_param,'ramp_pem')
model.ramp_com = assign_var(df_param,'ramp_com') # Not inluded in df_param
model.P_PV_cap = assign_var(df_param,'P_PV_cap') # not inlcuded - problem=
model.PT = assign_var(df_param,'PT') # proucer tariff 2021 EUR?
model.CT = assign_var(df_param,'CT') # consumer tariff 2021 EUR?
model.r_out = assign_var(df_param,'r_out') # exclude from V1 and V2
if version == 2 or version == 3:
    model.R_FCR_max = assign_var(df_param,'R_FCR_max') 
    model.R_FCR_min = assign_var(df_param,'R_FCR_min')
    model.bidres_FCR = assign_var(df_param,'bidres_FCR')
    model.R_aFRR_max = assign_var(df_param,'R_aFRR_max') #max bid size
    model.R_aFRR_min = assign_var(df_param,'R_aFRR_min') #min bid size 1 MW
    model.bidres_aFRR = assign_var(df_param,'bidres_aFRR') #100kW bid resolution
    model.R_mFRR_max = assign_var(df_param,'R_mFRR_max') #max bid size
    model.R_mFRR_min = assign_var(df_param,'R_mFRR_min') #min bid size 1 MW
    model.bidres_mFRR = assign_var(df_param,'bidres_mFRR') #100kW bid resolution
    model.c_FCR = pe.Param(model.Ω,model.T, initialize = c_FCRs) #implement Ω = 1 for model 2
    model.c_aFRR_up = pe.Param(model.Ω, model.T, initialize = c_aFRR_ups) #implement Ω = 1 for model 2
    model.c_aFRR_down = pe.Param(model.Ω, model.T, initialize = c_aFRR_downs) #implement Ω = 1 for model 2
    model.c_mFRR_up = pe.Param(model.Ω, model.T, initialize = c_mFRR_ups) #implement Ω = 1 for model 2
if version == 3:
    model.π_r = pe.Param(model.Ω, initialize = π_r)
    model.π_DA = pe.Param(model.Φ, initialize = π_DA)

# --------------------------- Defining model variables -----------------------------

model.p_PV = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.p_pem = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals, bounds=(0,model.P_pem_cap))
model.m_H2 = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals, bounds=(0,m_H2_max))
model.m_CO2 = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.m_Ri = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.s_raw = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.s_Pu = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals)
model.p_import = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals) #implement in V1
model.p_export = pe.Var(model.Ω, model.T, domain=pe.NonNegativeReals) #implement in V1
model.z_grid = pe.Var(model.Ω, model.T, domain = pe.Binary) #implement in V1

if version == 2 or version == 3:
    model.zFCR = pe.Var(model.T, domain = pe.Binary) #Defining the first binary decision variable
    model.zaFRRup = pe.Var(model.T, domain = pe.Binary) #binary decision variable
    model.zaFRRdown = pe.Var(model.T, domain = pe.Binary) #binary decision variable
    model.zmFRRup = pe.Var(model.T, domain = pe.Binary) #binary decision variable

    model.r_FCR =pe.Var(model.Ω, model.T, domain = pe.NonNegativeReals)
    model.r_aFRR_up = pe.Var(model.Ω, model.T, domain = pe.NonNegativeReals)
    model.r_aFRR_down = pe.Var(model.Ω, model.T, domain = pe.NonNegativeReals)
    model.r_mFRR_up = pe.Var(model.Ω, model.T, domain = pe.NonNegativeReals)

    # rx in model 2 - bx in model 3
    model.bx_FCR = pe.Var(model.T, domain = pe.NonNegativeIntegers)
    model.bx_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
    model.bx_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution
    model.bx_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeIntegers) #ancillary integer to realize the bid resolution

if version == 3:
    # Reserve bid variables
    model.b_FCR =pe.Var(model.T, domain = pe.NonNegativeReals) #Defining the variable of FCR reserve capacity
    model.b_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals)
    model.b_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeReals)
    model.b_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals)
    # Bid prices
    model.β_FCR = pe.Var(model.T, domain = pe.NonNegativeReals) #
    model.β_aFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals) #
    model.β_aFRR_down = pe.Var(model.T, domain = pe.NonNegativeReals) #
    model.β_mFRR_up = pe.Var(model.T, domain = pe.NonNegativeReals) #
    #bid acceptance binaries
    model.δ_FCR = pe.Var(model.Ω, model.T, domain = pe.Binary) #bid acceptance binary
    model.δ_aFRR_up = pe.Var(model.Ω, model.T, domain = pe.Binary) #bid acceptance binary
    model.δ_aFRR_down = pe.Var(model.Ω, model.T, domain = pe.Binary) #bid acceptance binary
    model.δ_mFRR_up = pe.Var(model.Ω, model.T, domain = pe.Binary) #bid acceptance binary



# ---------------- Objective function -----------------------------

if version == 1:
    expr = sum((model.c_DA[1,t]+model.CT)*model.p_import[1,t] - (model.c_DA[1,t]-model.PT)*model.p_export[1,t]  for t in model.T)
    model.objective = pe.Objective(sense = pe.minimize, expr=expr)


if version == 2:
    expr = sum((model.c_DA[1,t]+model.CT)*model.p_import[1,t] - (model.c_DA[1,t]-model.PT)*model.p_export[1,t] - (model.c_FCR[1,t]*model.r_FCR[1,t] + model.c_aFRR_up[1,t]*model.r_aFRR_up[1,t] + model.c_aFRR_down[1,t]*model.r_aFRR_down[1,t] + model.c_mFRR_up[1,t]*model.r_mFRR_up[1,t]) for t in model.T)
    model.objective = pe.Objective(sense = pe.minimize, expr=expr)

if version == 3:
    expr = sum(sum(model.π_r[ω]*(-(model.c_FCR[ω,t]*model.r_FCR[ω,t] + model.c_aFRR_up[ω,t]*model.r_aFRR_up[ω,t] + model.c_aFRR_down[ω,t]*model.r_aFRR_down[ω,t] + model.c_mFRR_up[ω,t]*model.r_mFRR_up[ω,t]) + sum(π_DA[φ]*((model.c_DA[φ,t]+model.CT)*model.p_import[ω,t] - (model.c_DA[φ,t]-model.PT)*model.p_export[ω,t]) for φ in model.Φ)) for ω in model.Ω) for t in model.T)
    model.objective = pe.Objective(sense = pe.minimize, expr=expr)


#-----------------------------------Power Flow Constraints-----------------------------------------------------------
model.c1 = pe.ConstraintList()
#Power balance constraint
for ω in model.Ω:
  for t in model.T:
    model.c1.add((model.p_import[ω,t]-model.p_export[ω,t]) + model.p_PV[ω,t] == model.p_pem[ω,t] + model.P_com)

model.c2 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c2.add(model.p_import[ω,t] <= model.z_grid[ω,t]*model.P_grid_cap)
    model.c2.add(model.p_export[ω,t] <= (1-model.z_grid[ω,t])*model.P_grid_cap)

model.c3 = pe.ConstraintList()
for ω in model.Ω: 
  for t in model.T:
    model.c3.add(model.p_PV[ω,t]<= model.P_PV_max[t])

model.c4 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c4.add(model.P_pem_min <= model.p_pem[ω,t])
    model.c4.add(model.p_pem[ω,t] <= model.P_pem_cap)


#---------------------------- Electrolyzer Efficiency (power consumption to hydrogen mass flow ratio) -----------------

# Piece-wise electrolyzer efficiency - has not been tested with stochastic model settings
if sEfficiency == 'pw':
  model.c_5 = Piecewise(model.T,
                          model.m_H2,model.p_pem,
                        pw_pts=pem_setpoint,
                        pw_constr_type='EQ',
                        f_rule=hydrogen_mass_flow,
                        pw_repn='SOS2')
                   
if sEfficiency == 'k':
  model.c5 = pe.ConstraintList()
  for ω in model.Ω:
    for t in model.T:
      model.c5.add(model.p_pem[ω,t] == model.m_H2[ω,t]/model.eff)

#------------------------------------------------Mass flow Constraints------------------------------------------------

#CO2 and Hydrogen mass flow fixed ratio (r_in)
model.c6 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c6.add(model.m_CO2[ω,t] == model.r_in*model.m_H2[ω,t])

model.c7 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c7.add(model.m_Ri[ω,t] == model.m_H2[ω,t] + model.m_CO2[ω,t])

model.c8 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c8.add(model.s_raw[ω,t] <= model.S_raw_max) 

#------------------------------------------------Storage level Constraints------------------------------------------------

# Define raw storage level as function of in- and out-flow
model.c9 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c9.add(model.s_raw[ω,t] == model.s_raw[ω,t-1] + model.m_Ri[ω,t] - model.m_Ro)

# Define 50% raw storage level prior to first simulation hour and at last simulation hour
model.c10 = pe.ConstraintList()
for ω in model.Ω:
  model.c10.add(model.s_raw[ω,1] == 0.5*model.S_raw_max + model.m_Ri[ω,1] - model.m_Ro)
  model.c10.add(0.5*model.S_raw_max == model.s_raw[ω,T])

# Define max storage level of pure methanol
model.c11 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    model.c11.add(model.s_Pu[ω,t] <= model.S_Pu_max)

# Storage level of pure methanol at time 1 equals the inflow in time 1 (empty tank prior to simulation)
model.c12 = pe.ConstraintList()
for ω in model.Ω:
  model.c12.add(model.s_Pu[ω,1] == model.m_Pu)

# Define raw storage level as function of in- and out-flow
model.c13 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c13.add(model.s_Pu[ω,t] == model.s_Pu[ω,t-1] + model.m_Pu - model.m_demand[t])

#----------------------------------------- ramp rate constraints -----------------------------------------------

model.c14 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c14.add(-model.ramp_pem * model.P_pem_cap <= model.p_pem[ω,t] - model.p_pem[ω,t-1])

model.c15 = pe.ConstraintList()
for ω in model.Ω:
  for t in model.T:
    if t >= 2:
      model.c15.add(model.p_pem[ω,t] - model.p_pem[ω,t-1] <= model.ramp_pem * model.P_pem_cap)


#----------------------------------------- reserve market constraints -----------------------------------------------

if version == 2 or version == 3:
    model.c16 = pe.ConstraintList()
    for t in model.T:
        model.c16.add(model.bx_FCR[t] >=(model.R_FCR_min/model.bidres_FCR)* model.zFCR[t])
        model.c16.add(model.bx_FCR[t] <=(model.R_FCR_max/model.bidres_FCR)* model.zFCR[t])
        model.c16.add(model.bx_aFRR_up[t] >= (model.R_aFRR_min/model.bidres_aFRR)*model.zaFRRup[t])
        model.c16.add(model.bx_aFRR_up[t] <= (model.R_aFRR_max/model.bidres_aFRR)*model.zaFRRup[t])
        model.c16.add(model.bx_aFRR_down[t] >= (model.R_aFRR_min/model.bidres_aFRR)*model.zaFRRdown[t])
        model.c16.add(model.bx_aFRR_down[t] <= (model.R_aFRR_max/model.bidres_aFRR)*model.zaFRRdown[t])
        model.c16.add(model.bx_mFRR_up[t] >= (model.R_mFRR_min/model.bidres_mFRR)*model.zmFRRup[t])
        model.c16.add(model.bx_mFRR_up[t] <= (model.R_mFRR_max/model.bidres_mFRR)*model.zmFRRup[t])

if version == 2:   
    model.c17 = pe.ConstraintList()
    for t in model.T:
        model.c17.add(model.r_FCR[1,t] == model.bx_FCR[t]*(model.bidres_FCR))
        model.c17.add(model.r_aFRR_up[1,t] == model.bx_aFRR_up[t]*(model.bidres_aFRR))
        model.c17.add(model.r_aFRR_down[1,t] == model.bx_aFRR_down[t]*(model.bidres_aFRR))
        model.c17.add(model.r_mFRR_up[1,t] == model.bx_mFRR_up[t]*model.bidres_mFRR)

    model.c21 = pe.ConstraintList()
    for t in model.T_block:
        model.c21.add(model.r_FCR[1,t+1] == model.r_FCR[1,t])
        model.c21.add(model.r_FCR[1,t+2] == model.r_FCR[1,t])
        model.c21.add(model.r_FCR[1,t+3] == model.r_FCR[1,t]) 


if version == 3:
    model.c18 = pe.ConstraintList()
    for t in model.T:
        model.c18.add(model.b_FCR[t] == model.bidres_FCR* model.bx_FCR[t])
        model.c18.add(model.b_aFRR_up[t] == model.bx_aFRR_up[t]*(model.bidres_aFRR))
        model.c18.add(model.b_aFRR_down[t] == model.bx_aFRR_down[t]*(model.bidres_aFRR))
        model.c18.add(model.b_mFRR_up[t] == model.bx_mFRR_up[t]*model.bidres_mFRR)

    model.c19 = pe.ConstraintList()
    M_FCR = max(c_FCRs.values()) # make sure that the correct series is applied (FCR / FCRs ?)
    M_aFRR_up = max(c_aFRR_ups.values()) 
    M_aFRR_down = max(c_aFRR_downs.values())
    M_mFRR_up = max(c_mFRR_ups.values())
    for ω in model.Ω:
        for t in model.T:
            model.c19.add(model.c_FCR[ω,t] - model.β_FCR[t] <= M_FCR*model.δ_FCR[ω,t])
            model.c19.add(model.c_aFRR_up[ω,t] - model.β_aFRR_up[t] <= M_aFRR_up*model.δ_aFRR_up[ω,t])
            model.c19.add(model.c_aFRR_down[ω,t] - model.β_aFRR_down[t] <= M_aFRR_down*model.δ_aFRR_down[ω,t])
            model.c19.add(model.c_mFRR_up[ω,t] - model.β_mFRR_up[t] <= M_mFRR_up*model.δ_mFRR_up[ω,t])
            model.c19.add(model.β_FCR[t] - model.c_FCR[ω,t] <= M_FCR * (1 - model.δ_FCR[ω,t]))
            model.c19.add(model.β_aFRR_up[t] - model.c_aFRR_up[ω,t] <= M_aFRR_up * (1 - model.δ_aFRR_up[ω,t]))
            model.c19.add(model.β_aFRR_down[t] - model.c_aFRR_down[ω,t] <= M_aFRR_down * (1 - model.δ_aFRR_down[ω,t]))
            model.c19.add(model.β_mFRR_up[t] - model.c_mFRR_up[ω,t] <= M_mFRR_up * (1 - model.δ_mFRR_up[ω,t]))


    model.c20 = pe.ConstraintList()
    for ω in model.Ω:
        for t in model.T:
            model.c20.add(model.r_FCR[ω,t] == model.b_FCR[t] * model.δ_FCR[ω,t])
            model.c20.add(model.r_aFRR_up[ω,t] == model.b_aFRR_up[t] * model.δ_aFRR_up[ω,t])
            model.c20.add(model.r_aFRR_down[ω,t] == model.b_aFRR_down[t] * model.δ_aFRR_down[ω,t])
            model.c20.add(model.r_mFRR_up[ω,t] == model.b_mFRR_up[t] * model.δ_mFRR_up[ω,t])


# --------------- Ensures that the FCR bid does not change for 4 consecutive hours -----------------
    model.c22 = pe.ConstraintList()
    for t in model.T_block:
        model.c22.add(model.b_FCR[t+1] == model.b_FCR[t])
        model.c22.add(model.b_FCR[t+2] == model.b_FCR[t])
        model.c22.add(model.b_FCR[t+3] == model.b_FCR[t]) 

    model.c23 = pe.ConstraintList()
    for t in model.T_block: 
        model.c23.add(model.β_FCR[t+1] == model.β_FCR[t]) 
        model.c23.add(model.β_FCR[t+2] == model.β_FCR[t]) 
        model.c23.add(model.β_FCR[t+3] == model.β_FCR[t]) 




# --------------------------grid constraints taking reserves into account--------------------------

if version == 2 or version == 3:
    # the possible increase in exports due to grid connection restrictions must surpass the supplied up-regulation capacity
    model.c24 = pe.ConstraintList()
    for ω in model.Ω:
        for t in model.T:
            model.c24.add(model.P_grid_cap + (model.p_import[ω,t]-model.p_export[ω,t])  >= model.r_FCR[ω,t] + model.r_aFRR_up[ω,t] + model.r_mFRR_up[ω,t])

    # the possible increase in imports, due to grid connection restrictions, must surpass the supplied down-regulation capacity
    model.c25 = pe.ConstraintList()
    for ω in model.Ω:
        for t in model.T:
            model.c25.add(model.P_grid_cap - (model.p_import[ω,t]-model.p_export[ω,t])  >= model.r_FCR[ω,t] + model.r_aFRR_down[ω,t])

    # the available increase in electrolyzer consumption must surpass the provided down-regulation
    model.c26 = pe.ConstraintList()
    for ω in model.Ω:
        for t in model.T:
            model.c26.add(model.P_pem_cap - model.p_pem[ω,t]  >= model.r_FCR[ω,t] + model.r_aFRR_down[ω,t])

    # the available decrease in electrolyzer consumption must surpass the provided up-regulation
    model.c27 = pe.ConstraintList()
    for ω in model.Ω:
        for t in model.T:
            model.c27.add(model.p_pem[ω,t] - model.P_pem_min >= model.r_FCR[ω,t] + model.r_aFRR_up[ω,t] + model.r_mFRR_up[ω,t])

if version == 3:
    # the bid of reserves must not surpass the full range of the electrolyzer loading
    model.c28 = pe.ConstraintList()
    for t in model.T:
        model.c28.add(model.b_FCR[t]*2 + model.b_aFRR_up[t] + model.b_aFRR_down[t] + model.b_mFRR_up[t] <= model.P_pem_cap - model.P_pem_min)


#-------------------------------------SOLVE THE MODEL-----------------------------------------------

instance = model.create_instance()
results = solver.solve(instance)
print(results)























Requirements to run the model: 
Install Python using Anaconda
Download the folder to a local path
Make sure to enable all VBA macro settings in Excel
 

To run model:  
1) Open Excel file "Opt_X.xlsm". When running the modedl the first time make sure to initialize environment.
2) Initialize virtual environment by pressing the "Initializ Environment" botton (This is only necessary first time).
	The terminal/cmd should open and start downloading and installing the necessary packages to run the model.  
3) Set parameters and choose model version.
4) Press the "Run Model" botton.
	The terminal/cmd should open and each operation is shown. When the model run is completed an Excel fil is created in the /Result files folder         

Model version options:
Model version = 1 for deterministic model and no reserve market included 
Model version = 2 for deterministic model with reserve markets(FCR, aFRR and mFRR) 
Model version = 3 for stochastic model with reserve markets(FCR, aFRR and mFRR) 


Simulation period: 
Should be full weeks i.e. 7,14,21... full days 

Note: 
Rember to save the Excel file before model run when changing parameters 





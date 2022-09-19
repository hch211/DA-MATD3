res_mg_1hour.pkl is the generation and demand data file of the Residential MEMG.

other_mg_da.pkl is the public order file of the Commercial and Industrial MEMGs.

res_da_env.py is the environment of Residential MEMG which includes the reward functions, 
system transition models and DA market clearing functions.

td3_da.py is the DA-MATD3 agent file which includes NN models and implementing idea of abstracting centrlised critic.

da_no_attack_main.py is the main file to set parameters and train the agent to find optimal policies (no attack occurs).

td3_mitigate.py is the TD3 agent file which is used to mitigate the attack.

mitigate_energy_trading.py is the main file to train the TD3 agent to find energy conversion policies to mitigate the FDIAs.

load_prediction.py is the main file to train the linear regression model for predict FDIAs. 
(This file need to be revised as it was used in a two-timestep energy trading and energy conversion model)

1h_model.h5 is the actor of the trained DA-MATD3 agent used for simulation in mitigate_energy_trading.py and load_prediction.py.

Useful codes is a folder of my other projects' codes for helping your revision which includes the environments 
of the commercial and industrial MEMG and generation and demand data of all three MEMGs.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_model(g10, g20, m1, m2, k, sigma_x, sigma_y, tmax, c=1, Noise=True):
    #Simulation parameters
    dt = 0.01
    t0 = 0
    tburn = 100

    #Initialise DataFrame for each variable to store all realisations
    df_sims_x = pd.DataFrame([])
    df_sims_y = pd.DataFrame([])

    # Initialise arrays to store single time-series data
    t = np.arange(t0,tmax,dt)
    g1 = np.zeros(len(t))
    g2 = np.zeros(len(t))


    # Model parameters
    if type(m1)==list:
        m1 = np.linspace(m1[0], m1[1], len(t))
    else:
        m1 = np.ones(int(tmax/dt))*m1

    if type(k)==list:
        k = np.linspace(k[0], k[1], len(t))
    else:
        k = np.ones(int(tmax/dt))*k


    def de_fun_g1(g1, g2, m1, k):
        return m1/(1+g2**2) - k*g1

    def de_fun_g2(g1, g2, m2, k):
        return m2/(1+c*g1**2) - k*g2

    # Create brownian increments (s.d. sqrt(dt))
    dW_x_burn = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = int(tburn/dt))
    dW_x = np.random.normal(loc=0, scale=sigma_x*np.sqrt(dt), size = len(t))

    dW_y_burn = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = int(tburn/dt))
    dW_y = np.random.normal(loc=0, scale=sigma_y*np.sqrt(dt), size = len(t))
    if Noise:
        factor=1
    else:
        factor=0
    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        g10 = g10 + de_fun_g1(g10, g20, m1[0], k[0])*dt + factor*dW_x_burn[i]
        g20 = g20 + de_fun_g2(g10, g20, m2, k[0])*dt + factor*dW_y_burn[i]

    # Initial condition post burn-in period
    g1[0]=1
    g2[0]=1
    
    matrices = []

    # Function to create 2x2 matrix with weighted combinations
    def create_matrix(g1, g2):
        return np.array([
            [1*g1 + 1*g2, 0.6*g1 + 0.4*g2],
            [0.4*g1 + 0.6*g2, 0.2*g1 + 0.8*g2]
        ])

    # Run simulation
    for i in range(len(t)-1):
        g1[i+1] = g1[i] + de_fun_g1(g1[i],g2[i], 3, k[i])*dt + factor*dW_x[i]
        g2[i+1] = g2[i] + de_fun_g2(g1[i],g2[i],m2, k[i])*dt + factor*dW_y[i]
        matrices.append(create_matrix(g1[i+1], g2[i+1]))

    # Store series data in a temporary DataFrame
    data ={
        'Time': t,
        'g1': g1,
        'g2': g2}
    df = pd.DataFrame(data)

    return df, np.array(matrices)

np.random.seed(2)

# As you see above I randomly choose those weights for the create_matrix function but by changing those we are able to have different combinations.
#%%
# here we can choose the type of dynamical systems from the code below and also change the parameters for it

data="pitchfork"#for generating the saddle trajectory(:"saddle")/pitchfork (:"pitchfork") trajectory data
dt = 0.01 #time difference used during simulation
tmax = 30 #total time duration for simulation
if data=="pitchfork":
    g10 = 0
    g20 = 0
    m1bif = 3.6
    m1start = 2.5
    m1end = m1start + (m1bif-m1start)*0.9
    m1 = [m1start, m1end]
    m2 = 1
    k = 1
    c=1
    sigma_x = 0.1
    sigma_y = 0.1

    df, matrices = simulate_model(g10, g20, m1, m2, k, sigma_x, sigma_y, tmax, c=c)
    # df = df.iloc[::100]
elif data=="saddle":
    g10 = 1
    g20 = 1
    m1 = 1
    m2 = 1
    # k = [1,1/4]
    k = [1*0.9, 0.9/4]
    kbif = 1/2
    c = 1

    sigma_x = 0.005
    sigma_y = 0.005

    df, matrices = simulate_model(g10, g20, m1, m2, k, sigma_x, sigma_y, tmax, c=c,)
    # df = df.iloc[::100]
    
plt.figure(figsize=(7, 3))
# Plot both columns on the same 
xticks = [tick / 100 for tick in range(0,len(df))]

# Plot matrix elements
plt.plot(xticks[:-1], matrices[:, 0, 0], label='Matrix [0,0]', linestyle='--')
plt.plot(xticks[:-1], matrices[:, 0, 1], label='Matrix [0,1]', linestyle='--')
plt.plot(xticks[:-1], matrices[:, 1, 0], label='Matrix [1,0]', linestyle='--')
plt.plot(xticks[:-1], matrices[:, 1, 1], label='Matrix [1,1]', linestyle='--')

# Customize the plot with labels and title
plt.xlabel('Time')  # or specify a different x-axis label
plt.ylabel('Expression')
plt.title('Generated Gene Expression over Time')  # or specify a different title
plt.legend()
plt
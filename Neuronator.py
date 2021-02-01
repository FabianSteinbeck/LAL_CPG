import numpy as np

# These functions setup the SNN and compute it
# builder: defines the neuron parameters
# computation: calculates the parameter changes within the timestep
# NetworkStepper: loops through the network
# weighting: defines the weights

def builder(preallocation) :

    NeuronIndices = tuple()
    NeuronIndices[0] = [1,2] # input neurons
    NeuronIndices[1] = [3,4] # speed neurons
    NeuronIndices[2] = [5,6] # Type II
    NeuronIndices[3] = [7,8] # Type I
    NeuronIndices[4] = [9,10] # Adaptation
    NeuronIndices[5] = [11,12] # Inhibition

    NP = dict()
    # Neuron Parameters
    # Membrane capacitance C_m
    NP['C_m'] = (1e-9, 0.5e-9, 0.5e-9, 0.5e-9, 0.5e-9, 0.5e-9) #[Farad]the smaller, the stronger excitability
    # Resting potential
    NP['V_rest'] = (-0.06, -0.06, -0.06, -0.06, -0.06, -0.06) # [V]
    # Cell potential V
    NP['V'] = np.zeros(12, preallocation) # [V]
    NP['V'][:,0] = NP['V_rest'][0] # Initial potential
    # Threshold potential V_th
    NP['V_th'] = (-0.05, -0.05, -0.05, -0.05, -0.05, -0.05) # [V]
    # Hyperpolarization potential V_hyper
    NP['V_hyper'] = (-0.065, -0.065, -0.065, -0.065, -0.065, -0.065) # [V]
    # Spike potential V_spike
    NP['V_spike'] = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02) # [V]
    # Injection current parameters - ------------------------------------------
    # Synaptic conductance G_injection
    NP['G_injection'] = (1.75e-9, 1.75e-9, 1.75e-9, 1.75e-9, 1.75e-9, 1.75e-9) # [Siemens] the lower, the lower excitability
    # Excitatory synaptic battery
    NP['V_injection_Ex'] = (0, 0, 0, 0, 0, 0) # [V]
    # Activation of synaptic conductance of the cell synapse for excitation
    NP['A_injection_Ex'] = np.zeros(12, preallocation)
    # Injection current excitatory
    NP['I_injection_Ex'] = np.zeros(12, preallocation) # [A]
    # Time constant for excitatory activation of synaptic conductance
    NP['Tau_injection_Ex'] = (0.02, 0.02, 0.02, 0.02, 0.02, 0.02) # [s]
    # Inhibitory synaptic battery
    NP['V_injection_In'] = (-0.08, -0.08, -0.08, -0.08, -0.08, -0.08) # [V]
    # Activation of synaptic conductance of the cell synapse for inhibition
    NP['A_injection_In'] = np.zeros(12, preallocation)
    # Time constant for inhibitory activation of synaptic conductance
    NP['Tau_injection_In'] = (0.03, 0.03, 0.03, 0.03, 0.03, 0.03) # [s]
    # Injection current inhibitory
    NP['I_injection_In'] = np.zeros(12, preallocation) # [A]
    # Spontaneous firing rate activation
    NP['A_spontaneous'] = (0, 0.07, 0.0315, 0.06, 0, 0)
    # Synaptic adaptation parameters - ----------------------------------------
    # synaptic adaptation conductance change per activation
    NP['G_adaptation'] = (0, 0, 0, 0, X, 0) # [Siemens]
    # adaptation conductance step
    NP['G_adaptation_Step'] = (0, 0, 0, 0, X, 0)
    # spike rate adaptation activation
    NP['A_adaptation'] = np.zeros(12, preallocation)
    # spike rate adaptation activation power
    NP['A_adaptation_P'] = (1, 1, 1, 1, X, 1)
    # adaptation battery for hyperpolarization
    NP['V_adaptation'] = (0, 0, 0, 0, -0.07, 0) # [V]
    # spike rate adaptation time constant
    NP['Tau_adaptation'] = (9999999999, 9999999999, 9999999999, 9999999999, X, 9999999999) # [s], huge values for excitatory neurons which don't adapt
    # synaptic input current
    NP['I_adaptation'] = np.zeros(12, preallocation)
    # Leak parameters - -------------------------------------------------------
    # Leak current
    NP['I_leak'] = np.zeros(12, preallocation) # [A]
    # Leak conductance
    NP['G_leak'] = (5e-9, 5e-9, 5e-9, 5e-9, 5e-9, 5e-9) # [Siemens]
    # spike events
    NP['spike'] = np.zeros(12, preallocation) # [Binary]

    return NP, NeuronIndices

# -----------------------------------------------------------------------------------------------------
def computation(t,input,N,dt,population,neuronIndex) :
    # Parameters: Inputs
    # t = timestep
    # input = current spikes of the whole network
    # Neuron = all neuron parameters
    # dt = time resolution
    # population = which of the network population is the current calculation
    # --> difference between inhibitory/excitatory
    # neuronIndex = which neuron of the whole network

    # first test if the neuron spikes now and change Activations accordingly
    if N['spike'][neuronIndex,t] == 1: #% did this neuron spike last time?
        # do not take external inputs
    else: # it didn't, take external inputs
        # set excitatory injection activation
        N['A_injection_Ex'][neuronIndex,t] = \
            N['A_injection_Ex'][neuronIndex,t] \
            + sum(input(input > 0)) \
            + N['A_spontaneous'][population]
        # set inhibitory injection activation
        N['A_injection_In'][neuronIndex,t] =  \
            N['A_injection_In'][neuronIndex,t] \
            + abs(sum(input(input < 0))) # find inhibitory input (everything negative)

    # Currents
    N['I_injection_Ex'][neuronIndex,t] = \
        N['G_injection'][population]*\
        N['A_injection_Ex'][neuronIndex,t]*\
        (N['V_injection_Ex'][population] \
         - N['V'][neuronIndex,t])

    N['I_injection_In'][neuronIndex,t] = \
        N['G_injection'][population]*\
        N['A_injection_In'][neuronIndex,t]*\
        (N['V_injection_In'][population]\
         - N['V'](neuronIndex,t))

    N['I_leak'][neuronIndex,t] = \
        N['G_leak'](population)*\
        (N['V_rest'](population) - \
         N['V'][neuronIndex,t])

    N['I_adaptation'][neuronIndex,t] = \
        N['G_adaptation'][population]*\
        N['A_adaptation'][neuronIndex,t]\
        ^N['A_adaptation_P'][population]*\
        (N['V_adaptation'][population] \
         - N['V'][neuronIndex,t])

    # Calculate new voltage, depending on if neuron spikes
    if N['spike'][neuronIndex,t] == 1: # yes
        # let it spike
        N['V'][neuronIndex,t+1] = N['V_spike'][population]
    elif N['V'][neuronIndex,t] == N['V_spike'][population]:
        # reset voltage to hyperpolarization in the next step
        N['V'][neuronIndex,t+1] = N['V_hyper'][population]
    else: # no
        # voltage change
        dV = (dt/N.C_m[population])*\
            (N['I_leak'][neuronIndex,t]\
            + N['I_injection_Ex'][neuronIndex,t]\
            + N['I_injection_In'][neuronIndex,t]\
            + N['I_adaptation'][neuronIndex,t])
        # add noise
        dV = dV + dV*0.275*rand(1);
        N['V'][neuronIndex,t+1] = N['V'][neuronIndex,t] + dV

    # synaptic activation variable A: decay
    # excitatory
    dA_syn_ex = -(dt/N['Tau_injection_Ex'][population])*\
               N['A_injection_Ex'][neuronIndex,t]
    N['A_injection_Ex'][neuronIndex,t+1] =\
        N['A_injection_Ex'][neuronIndex,t] + dA_syn_ex
    # inhibitory
    dA_syn_in = -(dt/N['Tau_injection_In'][population])*\
               N['A_injection_In'][neuronIndex,t]
    N['A_injection_In'][neuronIndex,t+1] =\
        N['A_injection_In'][neuronIndex,t] + dA_syn_in

    # conductance change da
    da = -(dt/N['Tau_adaptation'][population])*N['A_adaptation'][neuronIndex,t]
    # new adaptation activation
    N['A_adaptation'][neuronIndex,t+1] = N['A_adaptation'][neuronIndex,t] + da

    # is the voltage surpassing the threshold for spiking next timestep
    if N['V'][neuronIndex,t+1] >= N['V_th'][population]:
        if N['V'][neuronIndex,t+1] == N['V_spike'][population]:
            # do nothing, no spike command
        else:
            # record a spike event
            N['spike'][neuronIndex,t+1] = 1
            # increase spike rate adaptation "activation"
            N['A_adaptation'][neuronIndex,t+1] =\
                N['A_adaptation'][neuronIndex,t+1]\
                + N['G_adaptation_Step'][population]

    elif N['V'][neuronIndex,t+1] < N['V_hyper'][population]:
        N['V'][neuronIndex,t+1] = N['V_hyper'][population]

    return N

# ----------------------------------------------------------------------------------------------------
def NetworkStepper(N,N_Indices,t,dt,f,W):

# neuron calculations
for i in np.size(N_Indices,1) :# loop through all populations
    if i == 0 :# 1st population
        for ii in i:N_Indices(i+1) :# each ii depicts one neuron of the population
            N = NeuuCom(t,W(:,ii).*f,N,dt,i,ii) # input is familiarity

    elif i == 1 :# 2nd population
        for ii in (N_Indices(i-1)+1):N_Indices(i):
            N = NeuuCom(t,W(:,ii).*N.spike(:,t),N,dt,i,ii)

    elif i == 2 :# 3rd population
        for ii in (N_Indices(i-1)+1):N_Indices(i):
            N = NeuuCom(t,W(:,ii).*N.spike(:,t),N,dt,i,ii)

    elif i == 3 :# 4th population
        for ii in (N_Indices(i-1)+1):N_Indices(i):
            N = NeuuCom(t,W(:,ii).*N.spike(:,t),N,dt,i,ii)

    elif i == 4 :# 5th population
        for ii in (N_Indices(i-1)+1):N_Indices(i):
            N = NeuuCom(t,W(:,ii).*N.spike(:,t),N,dt,i,ii)

    elif i == 5 :# 6th population
        for ii in (N_Indices(i-1)+1):N_Indices(i):
            N = NeuuCom(t,W(:,ii).*N.spike(:,t),N,dt,i,ii)

    return N

# ----------------------------------------------------------------------------------------------------
def Weighting(C):

    # each weight represents the strength of connection to that neuron
    # columns: each neuron of the populations
    # rows: connections from the other neurons
    # negative weights depict inhibitory connections

    # setup connections for each population individually
    WE21 = 1 # Weight external input to population 1 (E21)
    W122 = 4
    W123 = 0.8
    W124 = 4
    W225 = 3
    W423 = 0.71
    W526 = 4
    W622 = X
    W623 = X
    W624 = X
    Weights =(  [WE21, 0, W122, 0, W123, 0, W124, 0, 0, 0, 0, 0],\
                [0, WE21, 0, W122, 0, W123, 0, W124, 0, 0, 0, 0],\
                [0, 0, 0, 0, 0, 0, 0, 0, W225, 0, 0, 0],\
                [0, 0, 0, 0, 0, 0, 0, 0, 0, W225, 0, 0],\
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],\
                [0, 0, 0, 0, 0, W423, 0, 0, 0, 0, 0, 0],\
                [0, 0, 0, 0, W423, 0, 0, 0, 0, 0, 0, 0],\
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, W526, 0],\
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, W526],\
                [0, 0, 0, W622, 0, W623, 0, 0, 0, 0, 0, 0],\
                [0, 0, W622, 0, W623, 0, W624, W624, 0, 0, 0, 0])
    return Weights

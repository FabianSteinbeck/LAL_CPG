import numpy as np

'''
These functions setup the SNN and compute it
- builder: defines the neuron parameters
- computation: calculates the parameter changes within the timestep
- NetworkStepper: loops through the network
- weighting: defines the weights
- TI: muscle neuron for turning
- TII: muscle neuron for turning
- TIIS: muscle neuron for speed
'''

def builder(preallocation) :

    '''
    builder sets up the network's neuron characteristics
    :param preallocation = how many timesteps
    :return: N = all neuron parameters
    :return: NeuronIndices = the indices of the different neurons
    '''

    # input neurons, adapting inhibition neurons, output neurons

    NeuronIndices = ([0, 1], [2, 3], [4, 5])

    N = dict()
    # Neuron Parameters ------------------------------------------------------------
    # Membrane capacitance C_m
    N['C_m'] = (1e-9, 1e-9, 0.5e-9)  # [Farad]the smaller, the stronger excitability
    # Resting potential
    N['V_rest'] = (-0.06, -0.06, -0.06)  # [V]
    # Cell potential V
    N['V'] = np.zeros((6, preallocation + 1))  # [V]
    N['V'][:,0] = N['V_rest'][0]  # Initial potential
    # Threshold potential V_th
    N['V_th'] = (-0.05, -0.05, -0.05)  # [V]
    # Hyperpolarization potential V_hyper
    N['V_hyper'] = (-0.065, -0.065, -0.065)  # [V]
    # Spike potential V_spike
    N['V_spike'] = (0.02, 0.02, 0.02)  # [V]
    # Injection current parameters - ------------------------------------------
    # Synaptic conductance G_injection
    N['G_injection'] = (1.75e-9, 4.2e-9, 3.1415e-9) # [Siemens] the lower, the lower excitability
    # Excitatory synaptic battery
    N['V_injection_Ex'] = (0, 0, 0)  # [V]
    # Activation of synaptic conductance of the cell synapse for excitation
    N['A_injection_Ex'] = np.zeros((6, preallocation + 1))
    # Injection current excitatory
    N['I_injection_Ex'] = np.zeros((6, preallocation + 1))  # [A]
    # Time constant for excitatory activation of synaptic conductance
    N['Tau_injection_Ex'] = (0.02, 0.035, 0.02)  # [s]
    # Inhibitory synaptic battery
    N['V_injection_In'] = (-0.08, -0.08, -0.08)  # [V]
    # Activation of synaptic conductance of the cell synapse for inhibition
    N['A_injection_In'] = np.zeros((6, preallocation + 1))
    # Time constant for inhibitory activation of synaptic conductance
    N['Tau_injection_In'] = (0.03, 0.07, 0.03)  # [s]
    # Injection current inhibitory
    N['I_injection_In'] = np.zeros((6, preallocation + 1))  # [A]
    # Spontaneous firing rate activation
    N['A_spontaneous'] = (0, 0, 0.12)
    # Synaptic adaptation parameters - ----------------------------------------
    # synaptic adaptation conductance change per activation
    N['G_adaptation'] = (0, 0.5e-8, 0)  # [Siemens]
    # adaptation conductance step
    N['G_adaptation_Step'] = (0, 0.1, 0)
    # spike rate adaptation activation
    N['A_adaptation'] = np.zeros((6, preallocation + 1))
    # spike rate adaptation activation power
    N['A_adaptation_P'] = (1, 2, 1)
    # adaptation battery for hyperpolarization
    N['V_adaptation'] = (0, -0.07, 0)  # [V]
    # spike rate adaptation time constant
    N['Tau_adaptation'] = (9999999999, 0.5, 9999999999) # [s], huge values for excitatory neurons which don't adapt
    # synaptic input current
    N['I_adaptation'] = np.zeros((6, preallocation + 1))
    # Leak parameters - -------------------------------------------------------
    # Leak current
    N['I_leak'] = np.zeros((6, preallocation + 1))  # [A]
    # Leak conductance
    N['G_leak'] = (5e-9, 5e-9, 5e-9)  # [Siemens]
    # spike events
    N['spike'] = np.zeros((6, preallocation + 1))  # [Binary]

    return N, NeuronIndices

'''
# -----------------------------------------------------------------------------------------------------
def computation(t,input,N,dt,population,neuronIndex) :
    
    computation calculates the new voltage state of a neuron, considering Ex/In inputs and adaptation
    :param t = timestep
    :param input = current spikes of the whole network calculated with weights
    :param N = all neuron parameters
    :param dt = time resolution
    :param population = which of the network population is the current calculation
    # --> difference between inhibitory/excitatory
    :param neuronIndex = which neuron of the whole network
    :return N = see above
    
    # first test if the neuron spikes now and change Activations accordingly
    if N['spike'][neuronIndex,t] == 1:
        #did this neuron spike last time? do not take external inputs
        N['A_injection_Ex'][neuronIndex,t] = 0
        N['A_injection_In'][neuronIndex,t] = 0

    else: # it didn't, take external inputs
        # set excitatory injection activation
        N['A_injection_Ex'][neuronIndex,t] = \
            N['A_injection_Ex'][neuronIndex,t] \
            + np.sum(x for x in input if x > 0) \
            + N['A_spontaneous'][population]
        # set inhibitory injection activation
        N['A_injection_In'][neuronIndex,t] =  \
            N['A_injection_In'][neuronIndex,t] \
            + abs(np.sum(x for x in input if x < 0)) # find inhibitory input (everything negative)

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
         - N['V'][neuronIndex,t])

    N['I_leak'][neuronIndex,t] = \
        N['G_leak'][population]*\
        (N['V_rest'][population] - \
         N['V'][neuronIndex,t])

    N['I_adaptation'][neuronIndex,t] = \
        N['G_adaptation'][population]*\
        pow(N['A_adaptation'][neuronIndex,t], N['A_adaptation_P'][population]) \
        *(N['V_adaptation'][population] \
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
        dV = (dt/N['C_m'][population])*\
            (N['I_leak'][neuronIndex,t]\
            + N['I_injection_Ex'][neuronIndex,t]\
            + N['I_injection_In'][neuronIndex,t]\
            + N['I_adaptation'][neuronIndex,t])
        # add noise
        dV = dV + 0.01*np.random.rand(1)
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
            N['spike'][neuronIndex,t+1] = 0
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
'''
#-------------------------------------------------------------------------
def computation(t, input, N, dt, population, neuronIndex):
    '''
    computation calculates the new voltage state of a neuron, considering Ex/In inputs and adaptation
    :param t = timestep
    :param input = current spikes of the whole network calculated with weights
    :param N = all neuron parameters
    :param dt = time resolution
    :param population = which of the network population is the current calculation
    # --> difference between inhibitory/excitatory
    :param neuronIndex = which neuron of the whole network
    :return N = see above
    '''
    # print(input)
    if N['spike'][neuronIndex,t] == 1: # % did this neuron spike last time? do not take external inputs

        N['A_injection_Ex'][neuronIndex, t] = 0
        N['A_injection_In'][neuronIndex, t] = 0

    else: # it didn't, take external inputs
        # set excitatory injection activation,  find excitatory input (everything positive)
        #print('+', np.sum(input[input > 0]))
        N['A_injection_Ex'][neuronIndex,t] =\
            N['A_injection_Ex'][neuronIndex,t]\
            + np.sum(input[input > 0]) \
            + N['A_spontaneous'][population]

        #% set inhibitory injection activation
        N['A_injection_In'][neuronIndex,t] = \
            N['A_injection_In'][neuronIndex,t] \
            + abs(np.sum(input[input < 0]))  # find inhibitory input (everything negative)
        #print('-', np.sum(abs(input[input > 0])))
        if abs(np.sum(input[input < 0])) > 0:
            y = 1


    # Currents ---------------------------------------------------------------
    N['I_injection_Ex'][neuronIndex,t] = \
        N['G_injection'][population]*N['A_injection_Ex'][neuronIndex,t]* \
       (N['V_injection_Ex'][population] - N['V'][neuronIndex,t])
    N['I_injection_In'][neuronIndex,t] = \
        N['G_injection'][population]*N['A_injection_In'][neuronIndex,t]* \
       (N['V_injection_In'][population] - N['V'][neuronIndex,t])
    N['I_leak'][neuronIndex,t] = \
        N['G_leak'][population]*(N['V_rest'][population] - N['V'][neuronIndex,t])
    N['I_adaptation'][neuronIndex,t] = \
        N['G_adaptation'][population]*pow(N['A_adaptation'][neuronIndex,t], N['A_adaptation_P'][population])* \
        (N['V_adaptation'][population] - N['V'][neuronIndex,t])

    # Calculate new voltage, depending on if neuron spikes -------------------
    if N['spike'][neuronIndex,t] == 1: # yes
        # let it spike
        N['V'][neuronIndex,t+1] = N['V_spike'][population]
    elif N['V'][neuronIndex,t] == N['V_spike'][population]:
        # reset voltage to hyperpolarization in the next step
        N['V'][neuronIndex,t+1] = N['V_hyper'][population]
    else: # no
        # voltage change
        dV = (dt/N['C_m'][population])*\
            (N['I_leak'][neuronIndex,t]\
            + N['I_injection_Ex'][neuronIndex,t]\
            + N['I_injection_In'][neuronIndex,t]\
            + N['I_adaptation'][neuronIndex,t])
        # add noise
        dV = dV + dV*0.25*np.random.rand(1)
        N['V'][neuronIndex,t+1] = N['V'][neuronIndex,t] + dV

    # synaptic activation variable A: decay ----------------------------------
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

    # conductance change da --------------------------------------------------
    da = -(dt/N['Tau_adaptation'][population])*N['A_adaptation'][neuronIndex,t]
    # new adaptation activation
    N['A_adaptation'][neuronIndex,t+1] = N['A_adaptation'][neuronIndex,t] + da

    # is the voltage surpassing the threshold for spiking next timestep ------
    if N['V'][neuronIndex,t+1] >= N['V_th'][population]:
        if N['V'][neuronIndex,t+1] == N['V_spike'][population]:
            # do nothing, no spike command
            N['spike'][neuronIndex,t+1] = 0

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
#--------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
def NetworkStepper(N,NeuronIndices,t,dt,f,W) :

    '''
    :param N = Neuron parameters
    :param N_Indices = Neuron indices
    :param t = current time step
    :param dt = step size in [s]
    :param f = familiarity
    :param W = WeightMatrix
    :return N = see above
    '''

    # neuron calculations
    for i in range(0, 3) :# loop through all populations
        if i == 0 :# 1st population
            for ii in [0, 1] :# each ii depicts one neuron of the population

                input = np.multiply(W[ii],f)
                print(W[ii])
                print(f)
                print(input)
                N = computation(t,input,N,dt,i,NeuronIndices[i][ii]) # input is familiarity

        elif i > 0 :# 2nd or 3rd population
            for ii in [0, 1] :
                input = np.multiply(W[ii], N['spike'][0:6,t])
                print(W[ii])
                print(f)
                print(input)
                N = computation(t,input,N,dt,i,NeuronIndices[i][ii])

    return N

# ----------------------------------------------------------------------------------------------------
def Weighting():
    '''
    each weight represents the strength of connection to that neuron
    columns: each neuron of the populations
    rows: connections from the other neurons
    negative weights depict inhibitory connections
    :return Weights: Matrix, where each Column represents the input weights from all other neurons
    '''
    # setup connections for each population individually
    WE21 = 1 # Weight external input to population 1 (E21)
    W122 = 5 # Weight from pop 1 to pop 2 (122)
    W123 = 0.5
    W222 = -3
    W223 = -1
    '''
    Weights =(  [WE21, 0,    W122, 0,    W123, 0],\
                [0,    WE21, 0,    W122, 0,    W123],\
                [0,    0,    0,    W222, 0,    W223],\
                [0,    0,    W222, 0,    W223, 0],\
                [0,    0,    0,    0,    0,    0],\
                [0,    0,    0,    0,    0,    0])
    '''
    Weights = ( [WE21,0,0,0,0,0], \
                [0,WE21,0,0,0,0],\
                [W122,0,0,W222,0,0],\
                [0,W122,W222,0,0,0],\
                [W123,0,0,W223,0,0],\
                [0,W123,W223,0,0,0])

    return Weights

def Accelerator(ExInput, a, A_injectionE):
    '''
    model the behaviour of a muscle generating
    force[acceleration(a)] with a leaky, integrate and not fire neuron
    :param ExInput = spikes from the LAL -output neurons
    :param a = acceleration[Voltage]
    :param A_injection = activation of the "muscles"
    '''
    # behaviour variables
    # membrane conductance: excitability of the neuron
    C_m = 3e-7 # [Farad]
    # synaptic conductance
    G_injection = 1e-5 # [Siemens]
    # maximum acceleration[deg / sec]
    a_injection_Ex = 0.025 * 1.8 #[deg / msec]
    # time constant for A_synaptic excitatory
    Tau_injection_Ex = 0.012 # [s]
    dt = 0.001

    # Leak conductance
    G_leak = 5e-6 # [Siemens]
    # Resting potential
    a_rest = 0 # [deg / sec]

    #
    A_injectionE = A_injectionE + ExInput
    I_injectionE = G_injection * A_injectionE * (a_injection_Ex - a)
    I_leak = G_leak * (a_rest - a)

    #voltage change - -----------------------------------------------------
    da = (dt / C_m) * (I_leak + I_injectionE)

    # new voltage
    a = a + da

    # synaptic activation variable A_injection: decay excitatory
    dA_syn = -(dt / Tau_injection_Ex) * A_injectionE
    A_injectionE = A_injectionE + dA_syn

    return a, A_injectionE
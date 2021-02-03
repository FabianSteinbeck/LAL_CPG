function [data] = LAL(C)

%% Simulation parameters --------------------------------------------------
% temporal resolution of simulation
dt = 0.001; %[s]
% duration of simulation
T = 2; % [s]
% steps of simulation
steps = T/dt;
% Setting up the neurons' parameters
[N,Populations,N_Indices] = Neuronator(steps,C);
% Setting up the weights
[Weights] = Weighting(C); %WeightMatrix
% load lookup table for input neurons
lookup = load('InputTable');
% load lookup tables for output to force neurons, make them accessible
global GTI
temp = load('OutputTableTI');
GTI = temp.value;
global GTII
temp = load('OutputTableTII');
GTII = temp.value;

% "World" settings
Agent.Centre = [zeros(1,steps);zeros(1,steps)]'; %Centrepoint of the Agent
Phi = zeros(1,steps);

% muscle activity
Agent.Acceleration = zeros(2,steps); %rows: neurons 1,2
Agent.AA = zeros(2,steps); %AccelerationActivation

Agent.Turning = zeros(4,steps); %rows: neurons 5,6,7,8
Agent.TA = zeros(4,steps); %Turning acceleration

% test
F = [25,50,75,100];
Agent.F = zeros(N_Indices(end),steps);
Agent.F(1,1:steps) = lookup.table(F(C(8)));
Agent.F(2,1:steps) = lookup.table(F(C(9)));
%% Simulation
for t = 1:steps
       
    %positions of the different parts of the agent
    [Agent.LeftEye(t,:),...
    Agent.RightEye(t,:),...
    Agent.LeftMotor(t,:),...
    Agent.RightMotor(t,:)]...
    = Bodypositions(...
    Agent.Centre(t,:),...
    Phi(t));
    
    % give the visual input to the LAL-network
    N = NetworkStepper(N,N_Indices,t,dt,Agent.F(:,t),Weights);
    
    % give the LAL-output-neurons' to the "muscle neurons"
    % output neuron #3
    [Agent.Acceleration(1,t+1),Agent.AA(1,t+1)] =...
    TIIS(N.spike(3,t)+N.spike(4,t),Agent.Acceleration(1,t),Agent.AA(1,t));
    % output neuron #4
    [Agent.Acceleration(2,t+1),Agent.AA(2,t+1)] =...
    TIIS(N.spike(3,t)+N.spike(4,t),Agent.Acceleration(2,t),Agent.AA(2,t));
    % output neuron #5
    [Agent.Turning(1,t+1),Agent.TA(1,t+1)] =...
    TII(N.spike(5,t),Agent.Turning(1,t),Agent.TA(1,t));
    % output neuron #6
    [Agent.Turning(2,t+1),Agent.TA(2,t+1)] =...
    TII(N.spike(6,t),Agent.Turning(2,t),Agent.TA(2,t));
    % output neuron #7
    [Agent.Turning(3,t+1),Agent.TA(3,t+1)] =...
    TI(N.spike(8,t),Agent.Turning(3,t),Agent.TA(3,t));
    % output neuron #8
    [Agent.Turning(4,t+1),Agent.TA(4,t+1)] =...
    TI(N.spike(7,t),Agent.Turning(4,t),Agent.TA(4,t));
    
    % what is the new direction with which distance
    [Rotation(t),...
    Distance(t)] =...
        ReactionApproach(...
    Agent.Acceleration(:,t),...
    Agent.Turning(:,t));
    
    %Current angle (Phi) + NewAngle (Rotation) + Noise
    Phi(t+1) = Phi(t) + Rotation(t);
    
    Agent.Centre(t+1,:) = CentrePosition(Agent.Centre(t,:), Phi(t+1), Distance(t));
    
end

%% visualization

% for i = 1:(N_Indices(end))
%     subplot(N_Indices(end),1,i)
%     bar(N.spike(i,:))
%     s(i) = sum(N.spike(i,:));
% end
% pause()
% close
% 
% bar(s)
% pause()
% close
% 
% plot(Agent.Centre(:,1),Agent.Centre(:,2))
% axis equal
% pause()
% close
%% Data packing

data{1} = C;
data{2} = Phi;
data{3} = Agent.Centre;
data{4} = N.spike;
data{5} = Distance;

%% 
end
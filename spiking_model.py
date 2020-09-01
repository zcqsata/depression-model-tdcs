############################################################################################
#
#   Simulation code for "A computational model of Major Depression: The role of glutamate 
#     dysfunction on cingulo-frontal network dynamics" 
#     Ramirez-Mahaluf J.P., Roxin A., Mayberg H.S. and Compte A. Cerebral Cortex, 2015
#
############################################################################################


# This Python code requires the installation of Brian www.briansimulator.org

from brian import *
from numpy.fft import rfft,irfft
from scipy.io import savemat
import numpy

#Network parameters

NE=80                 # Excitatory neurons, for fast simulations in the article we use 80
NI=20                   # Inhibitory neurons, for fast simulations in the article we use 20

#Biophysical parameters
tauav = 2*ms            # tau AMPA decay on vACC, this parameter was used to simulate MDD. Mild MDD (2.5%) = 2.05; Moderate MDD (5%) = 2.1; Severe MDD (7.5%) = 2.15
tauad = 2*ms            # tau AMPA decay on dlPFC
taun = 100*ms          # tau NMDA decay
taux = 2*ms              # tau NMDA rise
taug = 10*ms            # tau GABA decay
Vt  =-50*mvolt           # spike threshold
Vr  =-55*mvolt           # reset value
Elv  =-70*mvolt         # resting potential ventral ACC, this parameter was used to simulate SSRI treatment.
El  =-70*mvolt           # resting potential dlPFC
Ven = 16.129*mV      
refE= 2*ms		 # refractory periods piramidal cell
refI= 1*ms                 # refractory period inhibitory cell
cmE= 500*pF             #capacitance piramidal cel   
cmI= 200*pF              #capacitance interneuron  
tauE =20*ms              #tau piramidal cel
tauI =10*ms               #tau interneuron 
alpha =0.5*kHz
S=1	#Connectivity sparsensess; S=1, all-to-all connectivity was used in the article; use S<1 for sparse random connectivity
N=100/NE                  # Factor for rescaling the weights according to the number of neurons 

tacs_start_time = 2000*ms
tacs_end_time = 3000*ms

#Connection parameters
wgEEN = 0.001761*(1/S)*N     #weight  excitatory to excitatory through NMDA
wgEEA = 0.0009454*(1/S)*N    #weight  excitatory to excitatory through AMPA
wgEIN = 0.0012*(1/S)*N         #weight  excitatory to inhibitory through NMDA
wgEIA = 0.0004*(1/S)*N         #weight  excitatory to inhibitory through AMPA
wgIEG = 0.005*(1/S)*N          #weight inhibitory to excitatory through GABA
wgIIG = 0.004865*(1/S)*N     #weight inhibitory to  inhibitory through GABA
wgEIA1 = 0.0004*(1/S)*N      #weight vACC excitatory to dlPFC inhibitory through NMDA
wgEIA2 = 0.0004*(1/S)*N      #weight  dlPFC excitatory to vACC excitatory through NMDA

#equations excitatory cell vACC 
eqsE1 = '''
dV/dt = (-gea*V-gen*V/(1+exp(-V/Ven)/3.57)-gi*(V+70*mV)-(V-Elv))/(tauE) + I/cmE  : volt
dgea/dt = -gea/(tauav)           : 1
dgi/dt = -gi/(taug)              : 1
dspre/dt = -spre/(taun)+alpha*xpre*(1-spre) : 1
dxpre/dt = -xpre/(taux)                     : 1
gen: 1
I: amp
'''
#equations inhibitory cell vACC 
eqsI1 = '''
dV/dt = (-gea*V-gen*V/(1+exp(-V/Ven)/3.57)-gi*(V+70*mV)-(V-El))/(tauI) + I/cmI   : volt
dgea/dt = -gea/(tauav)           : 1
dgi/dt = -gi/(taug)             : 1
dspre/dt = -spre/(taun)+alpha*xpre*(1-spre) : 1
dxpre/dt = -xpre/(taux)                     : 1
gen: 1
I: amp

'''
#equations excitatory cell dlPFC 
eqsE2 = '''
dV/dt = (-gea*V-gen*V/(1+exp(-V/Ven)/3.57)-gi*(V+70*mV)-(V-El))/(tauE) + I/cmE  : volt
dgea/dt = -gea/(tauad)           : 1
dgi/dt = -gi/(taug)             : 1
dspre/dt = -spre/(taun)+alpha*xpre*(1-spre) : 1
dxpre/dt = -xpre/(taux)                     : 1
gen: 1
I: amp

'''
#equations inhibitory cell dlPFC 
eqsI2 = '''
dV/dt = (-gea*V-gen*V/(1+exp(-V/Ven)/3.57)-gi*(V+70*mV)-(V-El))/(tauI) + I/cmI   : volt
dgea/dt = -gea/(tauad)           : 1
dgi/dt = -gi/(taug)             : 1
dspre/dt = -spre/(taun)+alpha*xpre*(1-spre) : 1
dxpre/dt = -xpre/(taux)                     : 1
gen: 1
I: amp
'''

#Populations of neurons: 
Pev = NeuronGroup(NE, model= eqsE1, threshold=Vt, reset= Vr, refractory=refE)      #vACC excitatory neurons 
Piv = NeuronGroup(NI, model= eqsI1, threshold=Vt, reset= Vr, refractory=refI)          #vACC inhibitory neurons

Ped = NeuronGroup(NE, model= eqsE2, threshold=Vt, reset= Vr, refractory=refE)       #dlPFC excitatory neurons
Pid = NeuronGroup(NI, model= eqsI2, threshold=Vt, reset= Vr, refractory=refI)          #dlPFC inhibitory neurons

#Connection NMDA:
selfnmda_v = IdentityConnection(Pev, Pev, 'xpre', weight=1.0) #NMDA connections, excitatory to excitatory neurons in vACC
selfnmda_d = IdentityConnection(Ped, Ped, 'xpre', weight=1.0) #NMDA connections, excitatory to excitatory neurons in dlPF

#Connections AMPA and GABA:
Ceeav = Connection(Pev, Pev, 'gea', structure='dense') #AMPA connections, excitatory to excitatory neurons in vACC 
Ceiav = Connection(Pev, Piv, 'gea', structure='dense') #AMPA connections, excitatory to inhibitory neurons in vACC
Ciev = Connection(Piv, Pev, 'gi', structure='dense') # GABA connections, inhibitory to excitatory neurons in vACC
Ciiv = Connection(Piv, Piv, 'gi', structure='dense') # GABA connections, excitatory to excitatory neurons in vACC

Ceead = Connection(Ped, Ped, 'gea', structure='dense')#AMPA connections, excitatory to excitatory neurons in dlPFC 
Ceiad = Connection(Ped, Pid, 'gea', structure='dense') #AMPA connections, excitatory to inhibitory neurons in dlPFC
Cied = Connection(Pid, Ped, 'gi', structure='dense')# GABA connections, inhibitory to excitatory neurons in dlPFC
Ciid = Connection(Pid, Pid, 'gi', structure='dense')# GABA connections, excitatory to excitatory neurons in dlPFC

Ceiav1 = Connection(Pev, Pid, 'gea' )#AMPA connections, excitatory neurons in vACC target inhibitory neurons in dlPFC
Ceiad1 = Connection(Ped, Piv, 'gea' )#AMPA connections excitatory neurons in dlPFC target inhibitory neurons in vACC

Ceeav.connect_random(Pev, Pev, S, weight=wgEEA)  #AMPA connections, excitatory to excitatory neurons in vACC  
Ceiav.connect_random(Pev, Piv, S, weight=wgEIA)  #AMPA connections, excitatory to inhibitory neurons in vACC
Ciev.connect_random(Piv, Pev, S, weight=wgIEG)  # GABA connections, inhibitory to excitatory neurons in vACC
Ciiv.connect_random(Piv, Piv, S, weight=wgIEG)  # GABA connections, excitatory to excitatory neurons in vACC

Ceead.connect_random(Ped, Ped, S, weight=wgEEA) #AMPA connections, excitatory to excitatory neurons in dlPFC 
Ceiad.connect_random(Ped, Pid, S, weight=wgEIA) #AMPA connections, excitatory to inhibitory neurons in dlPFC
Cied.connect_random(Pid, Ped, S, weight=wgIEG) # GABA connections, inhibitory to excitatory neurons in dlPFC
Ciid.connect_random(Pid, Pid, S,weight=wgIIG) # GABA connections, excitatory to excitatory neurons in dlPFC

Ceiav1.connect_random(Pev, Pid, S, weight=wgEIA1) #AMPA connections, excitatory neurons in vACC target inhibitory neurons in dlPFC
Ceiad1.connect_random(Ped, Piv, S, weight=wgEIA2) #AMPA connections excitatory neurons in dlPFC target inhibitory neurons in vACC


#NMDA synapses
E_nmda_v = asarray(Pev.spre)
E_nmda_d = asarray(Ped.spre)
E_gen_v = asarray(Pev.gen)
E_gen_d = asarray(Ped.gen)
I_gen_v = asarray(Piv.gen)
I_gen_d = asarray(Pid.gen)

#Calculate NMDA contributions
@network_operation(when='start')
def update_nmda():
 E_gen_v[:] = wgEEN/wgEEA * numpy.dot(E_nmda_v,Ceeav.W)
 I_gen_v[:] = wgEIN/wgEIA * numpy.dot(E_nmda_v,Ceiav.W)
 E_gen_d[:] = wgEEN/wgEEA * numpy.dot(E_nmda_d,Ceead.W)
 I_gen_d[:] = wgEIN/wgEIA * numpy.dot(E_nmda_d,Ceiad.W)
 
 
#@network_operation(when='start')
def inject_current():
   if (defaultclock.t>tacs_start_time)&(defaultclock.t <tacs_end_time):
        Pev.I = 0.00000000000*amp
        Piv.I = 0.000000000000*amp
        Ped.I = 0.00000000000*amp
        Pid.I = 0.000000000000*amp


#External noise:
extinput1E=PoissonGroup(NE,rates=1800*Hz)
extinput1I=PoissonGroup(NI,rates=1800*Hz)

input1_coE=IdentityConnection(extinput1E,Pev,'gea',weight=0.082708)
input1_coI=IdentityConnection(extinput1I,Piv,'gea',weight=0.081)

extinput2E=PoissonGroup(NE,rates=1800*Hz)
extinput2I=PoissonGroup(NI,rates=1800*Hz)

input2_coE=IdentityConnection(extinput2E,Ped,'gea',weight=0.082708)
input2_coI=IdentityConnection(extinput2I,Pid,'gea',weight=0.081)


#Sadnnes task, emotional signal to vACC

exttaskinput1_on=4500*ms
exttaskinput1_off=5525*ms
exttaskinput1E=PoissonGroup(100,rates=lambda t: (t>exttaskinput1_on)*(t<exttaskinput1_off)*800*Hz)

#taskinput1_coE=IdentityConnection(exttaskinput1E,Pev,'gea',weight=0.0955)

#exttaskinput2_on=50000*ms
#exttaskinput2_off=50250*ms
#exttaskinput2E=PoissonGroup(100,rates=lambda t: (t>exttaskinput2_on)*(t<exttaskinput2_off)*800*Hz)

#taskinput2_coE=IdentityConnection(exttaskinput2E,Pev,'gea',weight=0.0955)

#exttaskinput3_on=55000*ms
#exttaskinput3_off=55250*ms
#exttaskinput3E=PoissonGroup(80,rates=lambda t: (t>exttaskinput3_on)*(t<exttaskinput3_off)*800*Hz)

#taskinput3_coE=IdentityConnection(exttaskinput3E,Pev,'gea',weight=0.0955)

#Working memory task, cognitive signal to dlPFC
exttaskinput4_on=6000*ms
exttaskinput4_off=6525*ms
exttaskinput4E=PoissonGroup(100,rates=lambda t: (t>exttaskinput4_on)*(t<exttaskinput4_off)*800*Hz)

#taskinput4_coE=IdentityConnection(exttaskinput4E,Ped,'gea',weight=0.0955)

#exttaskinput5_on=65000*ms
#exttaskinput5_off=65250*ms
#exttaskinput5E=PoissonGroup(80,rates=lambda t: (t>exttaskinput5_on)*(t<exttaskinput5_off)*800*Hz)

#taskinput5_coE=IdentityConnection(exttaskinput5E,Ped,'gea',weight=0.0955)

#exttaskinput6_on=70000*ms
#exttaskinput6_off=70250*ms
#exttaskinput6E=PoissonGroup(80,rates=lambda t: (t>exttaskinput6_on)*(t<exttaskinput6_off)*800*Hz)

#taskinput6_coE=IdentityConnection(exttaskinput6E,Ped,'gea',weight=0.0955)

#Deep Brain Stimulation (DBS): 
#extinput3I=SpikeGeneratorGroup(1,c_[zeros(2597),linspace(0*ms,19996.9*ms,2597)])
#input3_coI=Connection(extinput3I,Piv,'gea',weight=0.03)


#Save files
Miv = SpikeMonitor(Piv)
Mev = SpikeMonitor(Pev)
Mid = SpikeMonitor(Pid)
Med = SpikeMonitor(Ped)

Mv=PopulationRateMonitor(Pev,bin=0.1*second)
Md=PopulationRateMonitor(Ped,bin=0.1*second)
Mvm=PopulationRateMonitor(Pev,bin=0.5*second)
Mdm=PopulationRateMonitor(Ped,bin=0.5*second)


spikes_Ev = FileSpikeMonitor(Pev,'spikes_E_vACC.dat',record=True)
spikes_Ed = FileSpikeMonitor(Ped,'spikes_E_dlPFC.dat',record=True)
spikes_Iv = FileSpikeMonitor(Piv,'spikes_I_vACC.dat',record=True)
spikes_Id = FileSpikeMonitor(Pid,'spikes_I_dlPFC.dat',record=True)

#run
run(20*second)



#plot
subplot(2,2,1)
raster_plot(Mev, title=' vACC')


subplot(2,2,2)
plot(Mv.times,Mv.rate,Mvm.times,Mvm.rate,'ro:')
title('vACC')
ylabel('firing rate (Hz)')
xlabel('time (s)')

subplot(2,2,3)
raster_plot(Med, title='dlPFC')

subplot(2,2,4)
plot(Md.times,Md.rate,Mdm.times,Mdm.rate,'ro:')
title('dlPFC')
ylabel('firing rate (Hz)')
xlabel('time (s)')

show()
spikes_Ev.close_file()
spikes_Ed.close_file()
spikes_Iv.close_file()
spikes_Id.close_file()

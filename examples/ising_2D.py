import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=22)
params = {'legend.fontsize': 22}
plt.rcParams.update(params)

colorArray = ['k','r','b','g']

def ising_energy(spins,J=1):
    energy = 0
    M, N = spins.shape
    for i in range(M):
        for j in range(N):
            spin_domain = spins[i,j]
            neighbors = spins[(i+1)%N, j] + spins[i,(j+1)%N] + spins[(i-1)%N,j] + spins[i,(j-1)%N]
            energy += neighbors*spin_domain
    return -energy/2. # double counting


def ising_magnetization(spins):
    return spins.mean()

def boltz_factor(e_final=0.0,e_initial = 1.0,k=1,T=1):
    exponent = -(e_final-e_initial) / k / T
    return np.exp(exponent)

def heat_capacity_calc(energy,energy2,T,boltz_constant=1):
    num = energy2 - energy*energy
    denom = boltz_constant * T *T
    return num/denom

def susceptibility_calc(M,M2,T,boltz_constant=1):
    num = M2 - M*M
    denom = boltz_constant * T
    return num/denom

def metropolis(temperature=1,J=1,boltz_constant=1,N_grid=10,N_steps=2000,burn_in=1000,save_chain = True):
    spins = random.choice([-1.0,1.0],size=(N_grid,N_grid))
    energy = ising_energy(spins,J=J)
    en = np.zeros(N_steps)
    en2 = np.zeros(N_steps)
    mags = np.zeros(N_steps)
    mags2 = np.zeros(N_steps)
    for j in range(N_steps+burn_in):
        spinIndex = random.randint(0,N_grid,size=(2))
        spinIndex = spinIndex[0],spinIndex[1]
        temp_spins = np.copy(spins)
        temp_spins[spinIndex] = -temp_spins[spinIndex]
        proposed_energy = ising_energy(temp_spins,J=J)
        if proposed_energy <= energy:
            energy = proposed_energy
            spins = np.copy(temp_spins)
        else:
            probability = boltz_factor(e_final = proposed_energy,e_initial =energy,k=boltz_constant,T=temperature)
            # print probability
            check = random.uniform(0,1)
            if check <= probability:
                energy = proposed_energy
                spins = np.copy(temp_spins)
        if j < burn_in:
            continue
        else:
            en[j-burn_in]=(energy/spins.size)
            en2[j-burn_in] = (energy/spins.size)*(energy/spins.size)
            mag = ising_magnetization(spins)
            mags[j-burn_in] = mag
            mags2[j-burn_in] = mag*mag
    if save_chain == True:
        return en,mags,en2,mags2
    else:
        return en.mean(),mags.mean(),en2.mean(),mags2.mean()


N = 10
fig, ax = plt.subplots(2,2,figsize=(12/2,9/2),sharex=True)
temps = np.arange(1,3,0.025)
energies = np.zeros(temps.size)
magnetizations = np.zeros(temps.size)
heat_capacities = np.zeros(temps.size)
susceptibilities = np.zeros(temps.size)

for index,t in enumerate(temps):
    en,mags,en2,mags2 = metropolis(N_steps=7500000,temperature=t,N_grid=N,save_chain=False,burn_in=10000)
    energies[index] = en
    magnetizations[index] = np.abs(mags)
    heat_capacities[index] = heat_capacity_calc(en,en2,t)
    susceptibilities[index] = susceptibility_calc(mags,mags2,t)


ax[0][0].plot(temps,energies,c='k')
ax[0][1].plot(temps,magnetizations,c='k')
ax[1][0].plot(temps,heat_capacities,c='k')
ax[1][1].plot(temps,susceptibilities,c='k')

ax[0][0].set_ylim(-2.1,0.1)
ax[0][0].set_yticks([-2,-1,0])
ax[0][0].set_ylabel("E [arb]")

ax[0][1].set_ylim(-0.1,1.1)
ax[0][1].set_ylabel("M [arb]")
ax[0][1].set_xlabel("Temperature [arb]")


#ticks,lim
ax[1][0].set_ylabel("$C_{V}$ [arb]")

#ticks, lim
ax[1][1].set_ylabel("$\chi$ [arb]")


plt.show()
fig.savefig("biggest_burn_in_plot.pdf",bbox_inches='tight')










    # print

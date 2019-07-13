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

def metropolis(temperature=1,J=1,boltz_constant=1,N_grid=10,N_steps=2000,burn_in=1000,save_chain = True):
    spins = random.choice([-1.0,1.0],size=(N_grid,N_grid))
    energy = ising_energy(spins,J=J)
    en = np.zeros(N_steps)
    mags = np.zeros(N_steps)
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
            mag = ising_magnetization(spins)
            mags[j-burn_in] = mag
    if save_chain == True:
        return en,mags
    else:
        return en.mean(),mags.mean()


N = 10
fig, ax = plt.subplots(2,1,figsize=(12/2,9/2),sharex=True)
temps = np.arange(0.1,5.0,0.1)
energies = np.zeros(temps.size)
magnetizations = np.zeros(temps.size)
for index,t in enumerate(temps):
    en,mags = metropolis(N_steps=1500000,temperature=t,N_grid=N,save_chain=False)
    energies[index] = en
    magnetizations[index] = np.abs(mags)
ax[0].plot(temps,energies,c='k')
ax[1].plot(temps,magnetizations,c='r')
# en *= 2**(index+1)
# mags *= 2**(index*1)


ax[0].set_ylim(-2.1,0.1)
ax[0].set_yticks([-2,-1,0])
ax[0].set_ylabel("E [arb]")

ax[1].set_ylim(-0.1,1.1)
ax[1].set_ylabel("M [arb]")
ax[1].set_xlabel("Temperature [arb]")

plt.show()
fig.savefig("plot.pdf",bbox_inches='tight')










    # print

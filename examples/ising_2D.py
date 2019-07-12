import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

colorArray = ['k','r','b','g']

def ising_energy(spins,J=1):
    energy = 0
    for i in range(len(spins)):
        for j in range(len(spins)):
            spin_domain = spins[i,j]
            neighbors = spins[(i+1)%N, j] + spins[i,(j+1)%N] + spins[(i-1)%N,j] + spins[i,(j-1)%N]
            energy += neighbors*spin_domain
    return -energy/2. # double counting


def ising_magnetization(spins):
    return spins.mean()

def boltz_factor(e_final=0.0,e_initial = 1.0,k=1,T=1):
    exponent = -(e_final-e_initial) / k / T
    return np.exp(exponent)

def metropolis(temperature=1,J=1,boltz_constant=1,N=10,N_steps=10000,burn_in=1000):
    spins = random.choice([-1.0,1.0],size=(N,N))
    energy = ising_energy(spins,J=J)
    en = []
    mags = []
    for j in range(N_steps+burn_in):
        spinIndex = random.randint(0,N,size=(2))
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
            en.append(energy/spins.size)
            mag = ising_magnetization(spins)
            mags.append(mag)
    return en,mags

N = 10

fig, ax = plt.subplots(2,1,figsize=(8,16))
for index,t in enumerate([1,2.27,5]):
    en,mags = metropolis(N_steps=1000,temperature=t)
    # en *= 2**(index+1)
    # mags *= 2**(index*1)
    ax[0].plot(en,c=colorArray[index])
    ax[1].plot(mags,c=colorArray[index])

# ax[0].set_ylim(-2,0)
# ax[1].set_ylim(-0.1,1.1)
plt.show()










# print

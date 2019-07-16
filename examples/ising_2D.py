import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import time as time

### set some global plot parameters that make things look nice

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=22)
params = {'legend.fontsize': 22}
plt.rcParams.update(params)

def ising_energy(spins,J=1):
    """
    Calculates ising model energy for a set of 2D spins using periodic boundary conditions
    positional args:
        spins (numpy array, 2D): a grid with entries of spin = +/- 1
    keyword:
        J (float, optional): magnetic coupling, J>0 for ferromagnets
    returns float = ising energy
    """
    energy = 0
    M, N = spins.shape  #does not have to be a square matrix
    for i in range(M):
        for j in range(N):
            central_spin= spins[i,j]
            neighbors = spins[(i+1)%M, j] + spins[i,(j+1)%N] + spins[(i-1)%M,j] + spins[i,(j-1)%N]
            energy += neighbors*central_spin
    return -energy/2.0 # double counting


def ising_magnetization(spins):
    """
    Calculates magnetization of the grid of spins
    positional args:
        spins (numpy array, 2D)
    returns float of average magnetization
    """
    return spins.mean()

def boltz_factor(e_final=0.0,e_initial = 1.0,k=1,T=1):
    """
    Calculates boltzman factor / probability for the model to be found with a given energy
    positional args:
        None
    keyword args:
        e_final (float,optional): final energy of proposed step
        e_initial (float,optional): initial energy proposed step
        k (float,optional) : boltzman constant, set to one
        T (float,optional) : temperature in weird units
    returns:
        float of probability
    """
    exponent = -(e_final-e_initial) / k / T
    return np.exp(exponent)

def heat_capacity_calc(energy,energy2,T,boltz_constant=1):
    """
    calculates heat capacity from energy accumulators
    positional args:
        energy (float): mean energy, from calculation
        energy2 (float): mean energy2, from calculation
        T (float): temperature in weird units
    keyword args:
        boltz_constant (float,optional): boltzman constant, set to one
    returns:
        float of heat capacity
    """
    num = energy2 - energy*energy
    denom = boltz_constant * T *T
    return num/denom

def susceptibility_calc(M,M2,T,boltz_constant=1):
    """
    calculates magnetic susceptibilities from magnetization accumulators
    positional args:
        M (float): mean energy, from calculation
        M2 (float): mean energy2, from calculation
        T (float): temperature in weird units
    keyword args:
        boltz_constant (float,optional): boltzman constant, set to one
    returns:
        float of susceptibility
    """
    num = M2 - M*M
    denom = boltz_constant * T
    return num/denom

def metropolis(temperature=1,J=1,boltz_constant=1,N_grid=10,N_steps=2000,burn_in=1000,save_chain = True):
    """
    workhorse function, initializes the spin orientation randomly and updates according to boltzman distribution using the Metropolis algorithm
    positional arguments:
        None
    keyword args:
        temperature (float,optional): temperature for the calculation in weird units
        J (float,optional): interaction coupling, for ferromagnets J>0
        boltz_constant (float,optional): boltzman constant, set to one
        N_grid (int,optional): number of spins in each dimension of the lattice
        N_steps (int,optional): number of steps to be taken by the walker during the calculation
        burn_in (int,optional): number of burn-in steps to be taken before markov chain steps are counted
        save_chain (bool, optional): save the markov chain data, or only accumulators. default is True
    """
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

if __name__ == "main":
    # generates plot of observables
    N = 10
    fig, ax = plt.subplots(2,2,figsize=(12/2,9/2),sharex=True)
    temps = np.arange(1,3,0.025)
    energies = np.zeros(temps.size)
    magnetizations = np.zeros(temps.size)
    heat_capacities = np.zeros(temps.size)
    susceptibilities = np.zeros(temps.size)

    for index,t in enumerate(temps):
        start = time.time()
        en,mags,en2,mags2 = metropolis(N_steps=125000*12,temperature=t,N_grid=N,save_chain=False,burn_in=10000)
        final = time.time()
        print("{1:d}/{2:d}, {0:2.2f} seconds".format(final-start, index, len(temps)))
        energies[index] = en
        magnetizations[index] = np.abs(mags)
        heat_capacities[index] = heat_capacity_calc(en,en2,t)
        susceptibilities[index] = susceptibility_calc(mags,mags2,t)
    ax[0][0].plot(temps,energies,c='k')
    ax[0][1].plot(temps,magnetizations,c='k')


    ax[1][0].plot(temps,heat_capacities*100,c='k')
    ax[1][1].plot(temps,susceptibilities*100,c='k')

    ax[0][0].set_ylim(-2.1,0.1)
    ax[0][0].set_yticks([-2,-1,0])
    ax[0][0].set_ylabel("E [arb]")

    ax[0][1].set_ylim(-0.1,1.1)
    ax[0][1].set_yticks([0,1])
    ax[0][1].set_ylabel("M [arb]")
    ax[1][0].set_xlabel("Temperature [arb]")


    #ticks,lim
    ax[1][0].set_ylabel("$C_{V}$ [100 arb]")
    ax[1][0].set_yticks([0,1,2,3])
    ax[1][0].set_xticks([1,2,3])

    #ticks, lim
    ax[1][1].set_ylabel("$\chi$ [100 arb]")
    ax[1][1].set_xlabel("Temperature [arb]")
    #ax[1][1].set_yticks([0,1,2,3])

    ax[1][1].set_xticks([1,2,3])

    fig.subplots_adjust(hspace=0.5,wspace=0.5)
    plt.show()
    fig.savefig("new_plot.pdf",bbox_inches='tight')

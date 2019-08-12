import numpy as np
import matplotlib.pyplot as plt
import random as random
import os.path as path

plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=22)
params = {'legend.fontsize': 22}
plt.rcParams.update(params)

def lennard_jones(distance,epsilon=1.0,sigma=1.0):
    """
    Calculates Lennard-Jones potential given an epsilon (depth) and sigma (range)
    positional args:
        distance (float): distance to calculate the potential
    keyword:
        epsilon (float, optional): depth of the interaction, default is 1.0
        sigma (float,optional): range of interaction, default is 1.0
    returns:
        float: value of the LJ potential
    """
    prefactor = 4.0 * epsilon
    first = np.power(sigma/distance,12)
    second = - np.power(sigma/distance,6)
    return prefactor * (first + second)

def lennard_jones_derivative(distance,epsilon=1.0,sigma=1.0):
    """
    Calculates derivative of the Lennard-Jones potential given an epsilon (depth) and sigma (range)
    positional args:
        distance (float): distance to calculate the potential
    keyword:
        epsilon (float, optional): depth of the interaction, default is 1.0
        sigma (float,optional): range of interaction, default is 1.0
    returns:
        float: value of the LJ derivative
    """
    prefactor = 24.0 * epsilon * np.power(1.0/distance,2)
    first = 2*np.power(sigma/distance,12)
    second = - np.power(sigma/distance,6)
    return prefactor *(first+second)


def calc_forces(reduced_positions,box_width,ndim,num_atoms,max_interaction_radius=3.0,epsilon=1.0,sigma=1.0):
    """
    Calculates forces/accelerations/potential energies for each particle in a ndim-dimensional array
    positional args:
        reduced_positions (numpy array, (num_atoms,ndim) ): positions of each particle in ndim-dimensions in fractions of box length
        box_width (float or float-like): width of box (right now, only for equal dimensions)
        ndim (int): number of dimensions in which the particles are free to move (right now, only implemented for ndim = 2)
        num_atoms (int) : number of atoms for the simulation
    keyword:
        max_interaction_radius (float, optional): maximum distance between particles for which we will compute the interaction. else, interaction is negligible
        epsilon (float,optional): depth of Lennard-Jones potential, default is 1.0
    returns:
        1: (num_atoms,ndim): vector forces on each particle
        2: float: total potential energy  for particles in arrangement reduced_positions
    """
    reduced_interparticle_displacement = np.zeros(ndim) #in terms of axes fraction
    interparticle_displacement = np.zeros(ndim) # in physical units

    potential_energy = np.zeros(num_atoms) #initialize potential energy array for each atom
    force = np.zeros(reduced_positions.shape) #initialize force array for each atom/dimension (same shape as positions array)

    weakest_potential = lennard_jones(max_interaction_radius,epsilon=epsilon,sigma=sigma) # set this as the zero of potential for particles very far away from one another

    for i in range(num_atoms-1):
        for j in range(i+1,num_atoms): #combinatoric trick to keep pairs counting only once
            reduced_interparticle_displacement = reduced_positions[i,:]-reduced_positions[j,:]
            ### periodic boundary conditions; if particles are more than a half-box away, then loop to the adjacent cell
            for dim in range(ndim):
                if (np.abs(reduced_interparticle_displacement[dim])>0.5):
                    reduced_interparticle_displacement[dim] = reduced_interparticle_displacement[dim] - np.copysign(1.0,reduced_interparticle_displacement[dim])


            ### convert from axes fraction to absolute units for the calculation of physical quantities
            interparticle_displacement = box_width*reduced_interparticle_displacement
            interparticle_distance_squared = np.dot(interparticle_displacement,interparticle_displacement)
            interparticle_distance = np.sqrt(interparticle_distance_squared)

            ### only calculate the forces for particles that are within the cutoff range; otherwise, potential contribution/force is zero -- we are calculating the potential relative to the cutuff range potential
            if(interparticle_distance < max_interaction_radius ):

                potential = lennard_jones(interparticle_distance,epsilon=epsilon,sigma=sigma) - weakest_potential
                potential_derivative = lennard_jones_derivative(interparticle_distance,epsilon=epsilon,sigma=sigma)


                ### each particle gets 1/2 of the potential
                potential_energy[i] = potential_energy[i]+potential/2.0
                potential_energy[j] = potential_energy[j]+potential/2.0

                force[i,:] = force[i,:]+potential_derivative*reduced_interparticle_displacement ## Newton 3
                force[j,:] = force[j,:]-potential_derivative*reduced_interparticle_displacement

            else:
                potential_energy[i] = potential_energy[i] + 0.0
                potential_energy[j] = potential_energy[j] + 0.0
    return force, np.sum(potential_energy)/num_atoms

def calc_temp(vel,box_width,ndim,num_atoms):
    """
    Calculates tempterature for given particle velocity array
    positional args:
        vel (numpy array, (num_atoms,ndim) ): velocities of each particle in ndim-dimensions in fractions of box length
        box_width (float or float-like): width of box (right now, only for equal dimensions)
        ndim (int): number of dimensions in which the particles are free to move (right now, only implemented for ndim = 2)
        num_atoms (int) : number of atoms for the simulation
    keyword:
        None
    returns:
        1: float: average kinetic energy for arrangements
        2: float: kinetic temperature; cf. equipartition theorem for ndim-dimensions
    """

    kinetic_energy = 0.0
    for i in range(num_atoms):
        real_vel = box_width*vel[i,:]
        kinetic_energy = kinetic_energy + real_vel.dot(real_vel)/2.0

    kinetic_energy_average = 1.0*kinetic_energy/num_atoms
    temperature = 2.0*kinetic_energy_average/ndim #cf. equipartition theorem for monatomic species

    return kinetic_energy_average,temperature

def independent_samples_from_grid(box_min,box_max,num_atoms=10,minimum_separation=1.0):
    """
    Draws (num_atoms) unique samples from a 2-dimensional grid with mesh length = 1.0 over the ranges (box_min,box_max) for each dimension. No repeats.
    positional args:
        box_min (float or float-like): minimum coordinate for an axis of the allowed volume (all are equal)
        box_max (float or float-like): maximum coordinate for an axis of the allowed volume (all are equal)
    keyword:
        num_atoms (int) : number of atoms for the simulation
        minimum_separation (float): minimum interparticle-distance for initial of particle positions. If we accidentally put them too close, we can get *very* large repulsions and the program can break. A book (which I do not have at present) recommends ~ 2^1/6 * sigma, I went with 1.0 and have had no problems. lowest I've tried is about 0.85
    returns:
        (numpy array, (num_atoms,ndim=2 for now)) positional coordinates of num_atoms in two dimensions in "real" physical units. need to generalize for 3d soon
    """

    n_mesh = int(np.abs(box_max-box_min)/minimum_separation)
    n_mesh = complex(n_mesh)
    X,Y = np.mgrid[box_min:box_max:n_mesh, box_min:box_max:n_mesh]
    xy = np.vstack((X.flatten(),Y.flatten())).T
    return np.array([i  for i in random.sample(xy, num_atoms)])

def initialize_positions(ndim=2,num_atoms=32,box_width=10.0,write_positions=False,minimum_separation=1.0):
    """
    Initializes the positions of the atoms in the simulation
    positional args:
        None
    keyword:
        ndim (int): dimensionality of the simulation. presently only can be current default =  2
        num_atoms (int) : number of atoms for the simulation. default is 32
        box_width (float) : width of the box in the simulation. default is 10.0
        write_positions (bool) : if true, writes out the initial positions to a text file. default is false.
        minimum_separation (float): minimum interparticle-distance for initial of particle positions. If we accidentally put them too close, we can get *very* large repulsions and the program can break. A book (which I do not have at present) recommends ~ 2^1/6 * sigma, I went with 1.0 and have had no problems.
    returns:
        (numpy array, (num_atoms,ndim=2 for now)) positional coordinates of num_atoms in two dimensions in axes fraction coordinates. need to generalize for 3d soon
    """
    print "Generating new initial positions for N = {0:d} atoms, {1:d}-dimensional box with length = {2:2.1f}".format(num_atoms,ndim,box_width)
    pos = independent_samples_from_grid(-box_width/2.0+minimum_separation/2.0,box_width/2.0-minimum_separation/2.0,num_atoms=num_atoms,minimum_separation=minimum_separation)
    pos /= box_width #normalize positions to axes fraction
    if write_positions == True:
        np.savetxt(fileName,pos)
    return pos


def initialize_velocity(initial_temp=1.0,num_atoms=32,ndim=2,box_width=10.0):
    """
    Initializes the velocities of the atoms in the simulation
    positional args:
        None
    keyword:
        initial_temp (float): initial temperature of the system; used to calculate the appropriate magnitude of the velocity vector
        num_atoms (int) : number of atoms for the simulation. default is 32
        ndim (int): dimensionality of the simulation. presently only can be current default =  2
        box_width (float) : width of the box in the simulation. default is 10.0
    returns:
        (numpy array, (num_atoms,ndim=2 for now)) velocity vectors for num_atoms in two dimensions in axes fraction coordinates. need to generalize for 3d soon
    """
    vel_magnitude = np.sqrt(ndim*initial_temp)
    vel_total = np.random.normal(vel_magnitude,vel_magnitude/10.0,size=num_atoms)
    vx = np.random.normal(vel_magnitude/2.0,vel_magnitude/10.0,size=num_atoms)
    out_of_bounds_index = np.where(np.abs(vx) > vel_magnitude)
    vx[out_of_bounds_index] = vel_total[out_of_bounds_index]
    vx = vx * np.random.choice([-1.0,1.0],size=num_atoms)

    vy = np.sqrt(vel_total*vel_total-vx*vx)
    vy *= np.random.choice([-1.0,1.0],size=num_atoms)

    vel = np.ones((num_atoms,ndim))
    vel[:,0] *= vx /box_width
    vel[:,1] *= vy /box_width

    return vel,vel_magnitude


def time_evolve(num_atoms=10,num_steps=10000,time_step=0.001,initial_temp=1.0,output_step=1000,epsilon=1.0,sigma=1.0,box_width=10.0,ndim=2,burn_in = 0,boltz_factor = 1.0, mass = 1.0,minimum_separation=1.0):
    """
    Performs simultation; evolves according to time_step starting with given/random arrangement of atoms and velocities scaled according to the initial temperature. Uses Velocity-Verlet algorithm for forward stability in integration of Newton 2.
    positional args:
        None
    keyword:
        num_atoms (int) : number of atoms for the simulation. default is 10
        num_steps (int) : number of steps to take in the simulation. default is 10k
        time_step (float): amount by which to move forward in time for each step. default is 1E-3
        initial_temp (float): initial temperature of the system; used to calculate the appropriate magnitude of the velocity vector
        ndim (int): dimensionality of the simulation. presently only can be current default =  2
        output_step (int) : print dynamical properties (energy, temperature) to console at each step multiple of this number. default is 1k
        epsilon (float): depth of lennard-jones potential. default is 1.0
        sigma (float): range of lennard-jones potential. default is 1.0
        box_width (float) : width of the box in the simulation. default is 10.0
        ndim (int): dimensionality of the simulation; default is 2
        burn_in (int): number of steps to take before recording energies, temperature, etc. default is 0
        boltz_factor (float): boltzmann factor, default is 1.0 in natural units
        mass (float) : mass of atoms, all equal presently; changes to this not currently supported
        minimum_separation (float): minimum interparticle-distance for initial of particle positions. If we accidentally put them too close, we can get *very* large repulsions and the program can break. A book (which I do not have at present) recommends ~ 2^1/6 * sigma, I went with 1.0 and have had no problems. lowest I've tried is about 0.85
    returns:
        (numpy array, (num_atoms,ndim=2 for now)) velocity vectors for num_atoms in two dimensions in axes fraction coordinates. need to generalize for 3d soon
    """

    # take burn_in extra steps
    num_steps += burn_in

    # initialize chains to record values for each step and atom
    kinetic_energy_average = np.ones(num_steps)
    potential_energy_average = np.ones(num_steps)
    temperature = np.ones(num_steps)
    position_chain = np.ones((num_atoms,num_steps,ndim))
    velocity_chain = np.ones((num_atoms,num_steps,ndim))

    # initialize position, velocity, and acceleration
    position = initialize_positions(box_width=box_width,ndim=ndim,num_atoms=num_atoms,minimum_separation=minimum_separation)
    velocity,velocity_mag = initialize_velocity(initial_temp=initial_temp,ndim=ndim,num_atoms=num_atoms,box_width=box_width)
    acceleration = np.random.normal(0,velocity_mag/20.0,size=(num_atoms,ndim))/box_width



    for k in range(0,num_steps):

        # save position and velocity into memory
        position_chain[:,k,:] = position
        velocity_chain[:,k,:] = velocity


        #update position according to acceleration vector
        position = position + time_step*velocity + time_step*time_step*acceleration / 2.0

        # do the first velocity half-update
        velocity = velocity + time_step*acceleration/2.0


        # now calculate the new forces and acceleration at the new positions
        force, potential_energy_average[k] = calc_forces(position,box_width,ndim,num_atoms,epsilon=epsilon,sigma=sigma) # Step 3
        acceleration  = force/mass


        # do final velocity half-update
        velocity = velocity + time_step*acceleration/2.0

        #calculate the kinetic energy and temperature
        kinetic_energy_average[k],temperature[k] = calc_temp(velocity,box_width,ndim,num_atoms)


        ### periodic boundary conditions; if particle strays outside of the box, move it back from its new adjacent cell

        outside_bounds = np.where(position[:,:] > 0.5)
        position[outside_bounds]=  position[outside_bounds] - 1.0
        outside_bounds = np.where(position[:,:] < -0.5)
        position[outside_bounds]= position[outside_bounds] + 1.0

        #print out update
        if(k%output_step==0):
            print "Step {0:d}/{1:d}".format(k,num_steps)
            print "Energy: {0:2.4E}\nTemperature:{1:2.4E}\n".format(kinetic_energy_average[k]+potential_energy_average[k],temperature[k])


    #only read out parameters for runs after burn_in
    indices = num_steps - burn_in
    return kinetic_energy_average[-indices:], potential_energy_average[-indices:], temperature[-indices:], position_chain[:,-indices:,:],velocity_chain[:,-indices:,:]


if __name__ == "main":

    # makes some plots, haven't had time to clean up yet

    box = 10
    n_atoms = 20
    steps = 1000

    plot_frequency = 100
    output_step = 100
    #
    kin, pot, temp, pos,vel = time_evolve(num_atoms = n_atoms, initial_temp = 1.0 ,output_step=output_step,  num_steps=steps,box_width = box
    ,time_step=0.01,minimum_separation=1.0) #0.003 for reference time step


    fig, ax = plt.subplots(2,2)
    ax[0,0].plot(kin[0:-1:plot_frequency],'r-')
    ax[0,0].plot(pot[0:-1:plot_frequency],'g-')
    ax[0,0].plot(pot[0:-1:plot_frequency]+kin[0:-1:plot_frequency],'k-.')
    lim = ax[0,0].get_ylim()
    ax[0,0].set_ylabel(r"$E_{P}, E_{TOT}, E_{K}$")
    ax[0,0].xaxis.set_label_position("top")
    ax[0,0].xaxis.tick_top()
    ax[0,1].set_ylabel(r"Init. Pos.")
    for index, atom in enumerate(pos[:,0,0]):
        ax[0,1].scatter(pos[index,0,0]*box,pos[index,0,1]*box)
    ax[0,1].set_xlim(-box/2.0,box/2.0)
    ax[0,1].set_ylim(-box/2.0,box/2.0)
    ax[0,1].xaxis.tick_top()
    ax[0,1].yaxis.tick_right()
    ax[0,1].yaxis.set_label_position("right")
    ax[0,1].xaxis.set_label_position("top")
    ax[1,0].plot(temp[0:-1:plot_frequency],'k-')
    ax[1,0].set_ylim(np.percentile(temp,1)*0.9,np.percentile(temp,99)*1.1)
    ax[1,0].set_ylabel(r"$T$")
    for index, atom in enumerate(pos[:,0,0]):
        ax[1,1].scatter(pos[index,-1,0]*box,pos[index,-1,1]*box)

    ax[1,1].set_xlim(-box/2.0,box/2.0)
    ax[1,1].set_ylim(-box/2.0,box/2.0)
    ax[1,1].yaxis.tick_right()
    ax[1,1].yaxis.set_label_position("right")
    ax[1,1].set_ylabel("Fin. Pos.")

    ax[1,0].set_xlabel("t [{0:d} steps]".format(plot_frequency))
    ax[1,1].set_xlabel("t [{0:d} steps]".format(plot_frequency))
    plt.show()
    # fig.savefig("observables_natoms_{0:d}.pdf".format(n_atoms),bbox_inches='tight')

    speeds = vel[:,:,0]**2 + vel[:,:,1]**2
    speeds = np.sqrt(speeds)

    all_speeds = [speeds[k,:] for k in range(len(speeds[:,0]))]

    all_speeds = np.array(all_speeds)

    all_speeds = all_speeds.flatten()
    # all_speeds = speeds

    bins,dx = np.linspace(0,10,500,retstep=True)

    fig2, ax2 = plt.subplots()
    ax2.hist(all_speeds*box,bins=bins,histtype='step',color='k',weights=[1.0/len(all_speeds) for j in all_speeds])
    ax2.set_xlabel("Speed [arb]")
    ax2.set_ylabel("Prob. Dens. [1/{0:2.2f}]".format(dx))
    plt.show()

    # fig2.savefig("speed_distribution_natoms_{0:d}.pdf".format(n_atoms),bbox_inches='tight')

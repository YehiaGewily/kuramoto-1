import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class Kuramoto:

    def __init__(self, coupling=1, dt=0.01, T=10, n_nodes=None, natfreqs=None):
        '''
        coupling: float
            Coupling strength. Default = 1. Typical values range between 0.4-2
        dt: float
            Delta t for integration of equations.
        T: float
            Total time of simulated activity.
            From that the number of integration steps is T/dt.
        n_nodes: int, optional
            Number of oscillators.
            If None, it is inferred from len of natfreqs.
            Must be specified if natfreqs is not given.
        natfreqs: 1D ndarray, optional
            Natural oscillation frequencies.
            If None, then new random values will be generated and kept fixed
            for the object instance.
            Must be specified if n_nodes is not given.
            If given, it overrides the n_nodes argument.
        '''
        if n_nodes is None and natfreqs is None:
            raise ValueError("n_nodes or natfreqs must be specified")

        self.dt = dt
        self.T = T 
        self.coupling = coupling

        if natfreqs is not None:
            self.natfreqs = natfreqs
            self.n_nodes = len(natfreqs)
        else:
            self.n_nodes = n_nodes
            self.natfreqs = np.random.normal(size=self.n_nodes)

    def init_angles(self):
        '''
        Random initial random angles (position, "theta").
        '''
        return 2 * np.pi * np.random.random(size=self.n_nodes)

    def derivative(self, angles_vec, t, adj_mat, coupling):
        '''
        Compute derivative of all nodes for current state, defined as

        dx_i    natfreq_i + k  sum_j ( Aij* sin (angle_j - angle_i) )
        ---- =             ---
         dt                M_i

        t: for compatibility with scipy.odeint
        '''
        assert len(angles_vec) == len(self.natfreqs) == len(adj_mat), \
            'Input dimensions do not match, check lengths'

        angles_i, angles_j = np.meshgrid(angles_vec, angles_vec)
        interactions = adj_mat * np.sin(angles_j - angles_i)  # Aij * sin(j-i)

        dxdt = self.natfreqs + coupling * interactions.sum(axis=0)  # sum over incoming interactions
        return dxdt

    def integrate(self, angles_vec, adj_mat):
        '''Updates all states by integrating state of all nodes'''
        # Coupling term (k / Mj) is constant in the integrated time window.
        # Compute it only once here and pass it to the derivative function
        n_interactions = (adj_mat != 0).sum(axis=0)  # number of incoming interactions
        coupling = self.coupling / n_interactions  # normalize coupling by number of interactions
        '''
        here is the change
        ''' 
        t = np.linspace(0, self.T, int(self.T/self.dt))

        timeseries = odeint(self.derivative, angles_vec, t, args=(adj_mat, coupling))
        return timeseries.T  # transpose for consistency (act_mat:node vs time)



    def run(self, adj_mat=None, angles_vec=None):
        if angles_vec is None:
            angles_vec = self.init_angles()

        timeseries = self.integrate(angles_vec, adj_mat)
        return timeseries


    @staticmethod
    def phase_coherence(angles_vec):
        coherence = np.abs(np.mean(np.exp(1j * angles_vec)))
        return coherence

    def mean_frequency(self, act_mat, adj_mat):
        '''
        Compute average frequency within the time window (self.T) for all nodes
        '''
        assert len(adj_mat) == act_mat.shape[0], 'adj_mat does not match act_mat'
        _, n_steps = act_mat.shape

        # Compute derivative for all nodes for all time steps
        dxdt = np.zeros_like(act_mat)
        for time in range(n_steps):
            dxdt[:, time] = self.derivative(act_mat[:, time], None, adj_mat)

        # Integrate all nodes over the time window T
        integral = np.sum(dxdt * self.dt, axis=1)
        # Average across complete time window - mean angular velocity (freq.)
        meanfreq = integral / self.T
        return meanfreq
    


    def calculate_psi_within_r_range(self, adj_mat=None, angles_vec=None, r_min=0.6, r_max=0.8):
        if angles_vec is None:
            angles_vec = self.init_angles()

        timeseries = self.integrate(angles_vec, adj_mat)
        psi_values = []
        r_values = []
        time_steps = []

        for t in range(timeseries.shape[1]):
            current_phase = timeseries[:, t]
            order_parameter = np.mean(np.exp(1j * current_phase))

            # Calculate r at each timestep using the phase_coherence function
            r_t = self.phase_coherence(current_phase)

            # Store psi and r if r is within the specified range
            if r_min <= r_t <= r_max:
                # Calculate psi at each timestep as the argument of the order parameter
                psi_t = np.angle(order_parameter)
                psi_values.append(psi_t)
                r_values.append(r_t)
                time_steps.append(t * self.dt)  # Store the actual time step

        return time_steps, r_values, psi_values
    
    
    def phase_portrait(self, timeseries):
        """
        Calculate the phase portrait by computing the angle (theta) and its derivative (angular velocity) for each time step.

        Parameters:
        timeseries: 2D array
            The timeseries data from the Kuramoto model simulation.

        Returns:
        (theta, theta_dot): tuple of ndarray
            The angles and their derivatives.
        """
        theta = np.arctan2(np.sin(timeseries), np.cos(timeseries))  # Calculate angles from sine
        theta_dot = np.gradient(theta, axis=1) / self.dt  # Calculate angular velocity
        return theta, theta_dot

    def calculate_centroid(self, theta, theta_dot):
        """
        Calculate the centroid of the phase portrait.

        Parameters:
        theta: ndarray
            The angles of the oscillators.
        theta_dot: ndarray
            The angular velocities of the oscillators.

        Returns:
        centroid: tuple
            The centroid (mean_x, mean_y) of the phase portrait.
        """
        mean_x = np.mean(theta)
        mean_y = np.mean(theta_dot)
        return mean_x, mean_y

    def shift_by_centroid(self, theta, theta_dot, centroid):
        """
        Shift the phase portrait by its centroid.

        Parameters:
        theta: ndarray
            The angles of the oscillators.
        theta_dot: ndarray
            The angular velocities of the oscillators.
        centroid: tuple
            The centroid (mean_x, mean_y) of the phase portrait.

        Returns:
        shifted_theta, shifted_theta_dot: tuple of ndarray
            The shifted angles and angular velocities.
        """
        shifted_theta = theta - centroid[0]
        shifted_theta_dot = theta_dot - centroid[1]
        return shifted_theta, shifted_theta_dot

    def histogram_phase_portrait(self, theta, theta_dot, bins=30):
        """
        Calculate and flatten the 2D histogram of the phase portrait.

        Parameters:
        theta: ndarray
            The angles of the oscillators.
        theta_dot: ndarray
            The angular velocities of the oscillators.
        bins: int or [int, int]
            The number of histogram bins for each dimension (x and y).

        Returns:
        h: ndarray
            The flattened 2D histogram of the phase portrait.
        """
        H, xedges, yedges = np.histogram2d(theta.ravel(), theta_dot.ravel(), bins=bins)
        h = H.flatten()
        return h



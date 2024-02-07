import numpy as np
import operators as ops
import rungekutta4 as rk4
import matplotlib.pyplot as plt

# Initial data
def f(x):
    return np.exp(-((x - 0.2)/0.05)**2)

def l2_norm(vec, h):
    return np.sqrt(h)*np.sqrt(np.sum(vec**2))

def run_simulation(mx=100, order=2, show_animation=True):
    """Solves the advection equation using finite differences
    and Runge-Kutta 4.
    
    Method parameters: 
    mx:     Number of grid points, integer > 15.
    order:  Order of accuracy, 2, 4, 6, 8, 10 or 12
    """

    # Model parameters
    c = 0.5 # wave speed
    T = 1 # end time
    xl = 0 # left boundary
    xr = 1 # right boundary
    
    # Space discretization
    xvec, hx = np.linspace(xl, xr, mx, retstep=True)
    
    if order == 2:
        H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_2nd(mx, hx)
    elif order == 4:
        H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_4th(mx, hx)
    elif order == 6:
        H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(mx, hx)
        
    # RHS matrix, using SAT to impose BC.
    D = -c*D1 - c*HI@e_l@e_l.T
    
    # uncomment to check eigenvalues
    # eigD = np.linalg.eigvals(D.toarray())
    # plt.figure()
    # plt.scatter(eigD.real,eigD.imag)
    # plt.show()
    
    # Define right-hand-side function
    def rhs(u):
        return D@u
    
    # Time discretization
    ht_try = 0.1*hx
    mt = int(np.ceil(T/ht_try) + 1) # round up so that (mt-1)*ht = T
    tvec, ht = np.linspace(0, T, mt, retstep=True)
    
    # Initialize time variable and solution vector
    t = 0
    u = f(xvec)
    
    # Initialize plot for animation
    if show_animation:
        fig, ax = plt.subplots()
        [line] = ax.plot(xvec, u, label='Approximation')
        ax.set_xlim([xl, xr])
        ax.set_ylim([-1, 1.2])
        title = plt.title(f't = {0:.2f}')
        plt.draw()
        plt.pause(1)
        
    
    # Loop over all time steps
    for tidx in range(mt-1):
        # Take one step with the fourth order Runge-Kutta method.
        u, t = rk4.step(rhs, u, t, ht)
    
        # Update plot every 50th time step
        if show_animation and (tidx % 50 == 0 or tidx == mt-2) : 
            line.set_ydata(u)
            title.set_text(f't = {t:.2f}')
            plt.draw()
            plt.pause(1e-8)
    
    u_exact = f(xvec - c*T)
    err = l2_norm(u - u_exact,hx)
    
    if show_animation:
        plt.plot(xvec,u_exact,'--')
        plt.show()
        
    return err
        
if __name__ == "__main__":

    # Run convergence study
    mvec = np.array([21,61,101,201,401])
    order_vec = np.array([2,4,6])
    errvec = np.zeros((mvec.size,order_vec.size))
    for order_idx,order in enumerate(order_vec):
        for m_idx,m in enumerate(mvec):
            errvec[m_idx,order_idx] = run_simulation(m,order,show_animation=False)
    
    # Compute convergence rates
    q = np.zeros((mvec.size,order_vec.size))
    for order_idx,order in enumerate(order_vec):
        for m_idx,m in enumerate(mvec[1:]):
            q[m_idx+1,order_idx] = -np.log(errvec[m_idx,order_idx]/errvec[m_idx+1,order_idx])/np.log(mvec[m_idx]/mvec[m_idx+1])
    
    # Print tables with errors and convergence rates
    for order_idx,order in enumerate(order_vec):
        print("--- Order: %d ---" % order)
        print("m\tlog10(err)\tconv")
        for idx in range(mvec.size):
            print("%d\t%.2f\t\t%.2f" % (mvec[idx],np.log10(errvec[idx,order_idx]),q[idx,order_idx]))       
        print("")
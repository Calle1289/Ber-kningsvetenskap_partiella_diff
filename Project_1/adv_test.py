import numpy as np
import operators as ops
import rungekutta4 as rk4
import matplotlib.pyplot as plt

# Initial data
def u0(x):
    return np.exp(-((x-0.2)/0.05)**2)

# l2 norm
def l2_norm(vec, h):
    return np.sqrt(h) * np.sqrt(np.sum(vec**2))

def run_simulation(m=50, order=2, check_eigvals=False, show_animation=False):
    """Solve the advection equation."""

    # Model parameters
    xl = 0 # left boundary
    xr = 1 # right boundary
    c = 0.5 # wave speed
    T = 1 # final time

    # Grid
    xvec = np.linspace(xl, xr, m+1)
    h = (xr - xl)/m

    # SBP operators
    if order == 2:
        H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_2nd(m+1, h)
    elif order == 4:
        H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_4th(m+1, h)
    elif order == 6:
        H,HI,D1,D2,e_l,e_r,d1_l,d1_r = ops.sbp_cent_6th(m+1, h)

    # Build discretization matrix
    D = -c*D1 - c*HI@e_l@e_l.transpose()

    # RHS function
    def rhs(v):
        return D@v

    # Check eigenvalues of D
    if check_eigvals:
        eigD = np.linalg.eigvals(D.toarray())
        plt.figure()
        plt.scatter(eigD.real, eigD.imag)
        plt.draw()
        plt.show()

    # Select time step
    dt_try = 0.1*h/c
    mt = int(np.ceil(T/dt_try))
    dt = T/mt

    # Initialize
    t = 0
    v = u0(xvec)

    # Setup plot
    if show_animation:
        fig, ax = plt.subplots()
        [line] = ax.plot(xvec, v)
        plt.draw()
        plt.pause(1)
        ax.set_title(f't = 0')

    # Loop over all time steps
    for tidx in range(mt):
        v, t = rk4.step(rhs, v, t, dt)

        if show_animation:
            line.set_ydata(v)
            ax.set_title(f't = {t:.2}')
            plt.draw()
            plt.pause(1e-8)

    if show_animation:
        plt.show()

    u_exact = u0(xvec - c*T)
    err_vec = v - u_exact
    l2_err = l2_norm(err_vec, h)
    return l2_err

def run_convergence():
    ms = [25,50,100,200,400]
    orders = [2, 4, 6]

    # Run all simulations
    err_dict = {}
    for o in orders:
        err_dict[o] = []
        for m in ms:
            err = run_simulation(m, o)
            err_dict[o].append(err)

    # Compute convergence rates
    q_dict = {}
    for o in orders:
        q_dict[o] = [0.0] # Set first rate to 0
        err_vec = err_dict[o]
        for i in range(len(err_vec)-1):
            q_dict[o].append(np.log(err_vec[i]/err_vec[i+1])/np.log(ms[i+1]/ms[i]))

    # Print
    for o in orders:
        print(f'----- Order: {o} ------')
        for m, err, q in zip(ms, err_dict[o], q_dict[o]):
            print(f'{m}\t{err:.2e}\t{q:.2f}')
                

def main():
    #run_simulation(50, 4, True)
    run_convergence()

if __name__ == '__main__':
    main()
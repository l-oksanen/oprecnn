import numpy as np

def solver(dt, T, f=None, v0=None, v1=None, h0=None, h1=None, 
    a=None, cmax=1, L=1, C=1, user_action=None):
    """
    Solve u_tt=a*u_xx + f on (0,T) x (0,L)
    with initial conditions u=v0 and ut=v1 at t=0
    and boundary conditions ux=h0 and ux=h1 at x=0 and x=L, respectively.
    Time step is given by dt, the Courant number by C, 
    and cmax is the maximum of the speed of sound i.e. max(sqrt(a)).
    Function user_action(u, t, x, n) is called at each time step.
    Here u is the solution at time t[n] in the mesh x.

    Returns u, t, x where u is the the solution at t=T in the mesh x 
    and t is the mesh in time. 
    """
    # Defaults
    f = (lambda t, x: 0) if f is None else f

    def init(param, val):
        if param is None:
            return (lambda x: val if np.isscalar(x) else val*np.ones_like(x)) 
        else:
            return param
    v0 = init(v0, 0)
    v1 = init(v1, 0)
    h0 = init(h0, 0)
    h1 = init(h1, 0)
    a = init(a, cmax**2)

    user_action = (lambda u, t, x, n: False) \
        if user_action is None else user_action

    # Meshes
    Nt = int(round(T/dt))
    t, dt = np.linspace(0, T, Nt+1, retstep=True)   # mesh in time
    Nx = int(round(C*L/(dt*cmax)))
    x, dx = np.linspace(0, L, Nx+1, retstep=True)   # mesh in space

    # Help variables in the scheme
    dt2 = dt**2   
    dd2 = dt2 / dx**2

    # Storage arrays 
    u_np1 = np.zeros(Nx+1)   # solution at n+1
    u_n   = np.zeros(Nx+1)   # solution at n
    u_nm1 = np.zeros(Nx+1)   # solution at n-1

    # At n=0 load initial condition 
    u_nm1 = v0(x)
    user_action(u_nm1, t, x, 0)

    # At n=1 use special formula 
    u_n[0] = u_nm1[0] + dt*v1(x[0]) + \
        dd2*a(x[0]) * (u_nm1[1] - u_nm1[0] - dx*h0(t[0])) + \
        0.5*dt2 * f(t[0], x[0]) 
    u_n[Nx] = u_nm1[Nx] + dt*v1(x[Nx]) + \
        dd2*a(x[Nx]) * (u_nm1[Nx-1] - u_nm1[Nx] + dx*h1(t[0])) + \
        0.5*dt2 * f(t[0], x[Nx])
    u_n[1:-1] = u_nm1[1:-1] + dt*v1(x[1:-1]) + \
        0.5*dd2*a(x[1:-1]) * (u_nm1[2:] - 2*u_nm1[1:-1] + u_nm1[:-2]) + \
        0.5*dt2 * f(t[0], x[1:-1])
    user_action(u_n, t, x, 1)

    # Compute u_np1 given u_n and u_nm1
    for n in range(1, Nt):
        u_np1[0] = -u_nm1[0] + 2*u_n[0] + \
            2*dd2*a(x[0]) * (u_n[1] - u_n[0] - dx*h0(t[n])) + \
            dt2 * f(t[n], x[0]) 
        u_np1[Nx] = -u_nm1[Nx] + 2*u_n[Nx] + \
            2*dd2*a(x[Nx]) * (u_n[Nx-1] - u_n[Nx] + dx*h1(t[n])) + \
            dt2 * f(t[n], x[Nx])
        u_np1[1:-1] = -u_nm1[1:-1] + 2*u_n[1:-1] + \
            dd2*a(x[1:-1]) * (u_n[2:] - 2*u_n[1:-1] + u_n[:-2]) + \
            dt2 * f(t[n], x[1:-1])
        if user_action(u_np1, t, x, n+1): break
        # Swap storage arrays for next step
        u_nm1, u_n, u_np1 = u_n, u_np1, u_nm1

    return u_n, t, x

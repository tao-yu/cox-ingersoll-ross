import numpy as np
from scipy.stats import ncx2
from numba import jit, njit


def solve_em(f, g, X_0, t, W):
    X_em = np.zeros(W.shape)
    
    X_em[0] = X_0
    X_temp = X_em[0]
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        X_temp = X_temp + dt*f(X_temp) + g(X_temp)*W_inc
        X_em[j] = X_temp
    return t, X_em


def solve_milstein(f, g, g_deriv, X_0, t, W):
    X_em = np.zeros(len(W))
    X_temp = X_0
    X_em[0] = X_0
    for j in range(1, len(W)):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        X_temp = X_temp \
                 + dt*f(X_temp) \
                 + g(X_temp) * W_inc \
                 + 0.5 * g(X_temp) * g_deriv(X_temp) * (W_inc**2 - dt)
        X_em[j] = X_temp
    return t, X_em


@njit
def implicit_scheme(k, lamda, theta, X_0, t, W):
    alpha = (4*k*lamda - theta**2)/8
    beta = -k/2
    gamma = theta/2

    Y_sol = np.zeros(W.shape)
    
    Y_sol[0] = np.sqrt(X_0)
    Y_temp = Y_sol[0]
    #set_trace()
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        discriminant = (Y_temp + gamma*W_inc)**2/(4*(1-beta*dt)**2) \
                       + (alpha*dt)/(1-beta*dt)

        Y_temp = (Y_temp + gamma*W_inc)/(2*(1-beta*dt)) \
                 + np.sqrt(
                     np.abs(discriminant)
                 )
        Y_temp[discriminant < 0] = 0
        Y_sol[j] = Y_temp
    X_sol = Y_sol**2
    return t, X_sol


@njit
def explicit_scheme(k, lamda, theta, X_0, t, W, l=0):
    X_sol = np.zeros(W.shape)
    X_sol[0] = np.sqrt(X_0)
    X_temp = X_sol[0]
    
    dt = t[1] - t[0]
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        #dt = t[j] - t[j-1]
        X_temp = ((1-dt*k/2)*np.sqrt(X_temp) + (theta*W_inc)/(2*(1-dt*k/2)))**2 + (lamda*k-theta**2/4)*dt + l*((W_inc)**2 - dt)
        X_temp[X_temp<0] = 0
        X_sol[j] = X_temp
    return t, X_sol


@njit
def deelstra_delbaen(k, lamda, theta, X_0, t, W):
    cir_drift = lambda x: k*(lamda - x)
    cir_diff = lambda x: theta*np.sqrt(np.maximum(x, 0))

    X_em = np.zeros(W.shape)
    
    X_em[0] = X_0
    X_temp = X_em[0]
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        X_temp = X_temp + dt*cir_drift(X_temp) + cir_diff(X_temp)*W_inc
        X_em[j] = X_temp
    return t, X_em


@njit
def diop(k, lamda, theta, X_0, t, W):
    cir_drift = lambda x: k*(lamda - x)
    cir_diff = lambda x: theta*np.sqrt(x)

    X_em = np.zeros(W.shape)
    
    X_em[0] = X_0
    X_temp = X_em[0]
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        X_temp = np.abs(X_temp + dt*cir_drift(X_temp) + cir_diff(X_temp)*W_inc)
        X_em[j] = X_temp
    return t, X_em


@njit
def euler(k, lamda, theta, X_0, t, W):
    X_em = np.zeros(W.shape)
    X_em[0] = X_0
    X_temp = X_em[0]
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        X_temp = X_temp + k*(lamda - X_temp)*dt + theta*np.sqrt(X_temp)*W_inc
        X_em[j] = X_temp
    return t, X_em


@njit
def higham_mao(k, lamda, theta, X_0, t, W):
    X_em = np.zeros(W.shape)
    X_em[0] = X_0
    X_temp = X_em[0]
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        X_temp = X_temp + k*(lamda - X_temp)*dt + theta*np.sqrt(np.abs(X_temp))*W_inc
        X_em[j] = X_temp
    return t, X_em


@njit
def full_truncation(k, lamda, theta, X_0, t, W):
    pos = lambda x: np.maximum(0, x)
    X_em = np.zeros(W.shape)
    X_em[0] = X_0
    X_temp = X_em[0]
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        X_temp = X_temp + k*(lamda - pos(X_temp))*dt + theta*np.sqrt(pos(X_temp))*W_inc
        X_em[j] = X_temp
    return t, pos(X_em)


@njit
def absorption(k, lamda, theta, X_0, t, W):
    pos = lambda x: np.maximum(0, x)
    X_em = np.zeros(W.shape)
    X_em[0] = X_0
    X_temp = X_em[0]
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        X_temp = pos(pos(X_temp) + k*(lamda - pos(X_temp))*dt + theta*np.sqrt(pos(X_temp))*W_inc)
        X_em[j] = X_temp
    return t, X_em
    

@njit
def bossy(k, lamda, theta, X_0, t, W):
    X_em = np.zeros(W.shape)
    X_em[0] = X_0
    X_temp = X_em[0]
    for j in range(1, W.shape[0]):
        W_inc = W[j] - W[j-1]
        dt = t[j] - t[j-1]
        X_temp = np.abs(X_temp) + k*(lamda - X_temp)*dt + theta*np.sqrt(X_temp)*W_inc
        X_em[j] = X_temp
    return t, X_em


def direct_simulation(k, lamda, theta, X_0, T, N):
    dt = T/N
    c = (2*k)/((1-np.exp(-k*dt))*theta**2)
    df = 4*k*lamda/theta**2
    X_temp = X_0
    X_sol = np.zeros(N+1)
    X_sol[0] = X_0
    for j in range(1, N+1):
        nc = 2*c*X_temp*np.exp(-k*dt)
        Y = ncx2.rvs(df, nc, size=1)
        X_temp = Y/(2*c)
        X_sol[j] = X_temp
    return np.linspace(0, T, N+1), X_sol


@njit
def hybrid_adaptive(k, lamda, theta, X_0, T, h_max, h_min, r):
    alpha = (4*k*lamda - theta**2)/8
    beta = -k/2
    gamma = theta/2
    
    max_steps = int(T//h_min + 1)
    t = np.zeros(max_steps)
    Y = np.zeros(max_steps)
    std_n = np.random.randn(max_steps)
    
    backstop_usage = np.zeros(max_steps)
    
    Y[0] = np.sqrt(X_0)

    def backstop(Y, dt, dW):
        discriminant = (Y + gamma*dW)**2/(4*(1-beta*dt)**2) \
                    + (alpha*dt)/(1-beta*dt)
        if discriminant < 0:
            return 0
        else:
            return (Y + gamma*dW)/(2*(1-beta*dt)) \
                     + np.sqrt(
                         np.abs(discriminant)
                     )
    
    def explicit_step(Y, dt, dW):
        return Y + dt*(alpha/Y + beta*Y) + gamma * dW
    
    
    for j in range(1, max_steps):
        h = h_max * np.minimum(1, np.abs(Y[j-1])**r)
        if t[j-1] + h > T:
            t[j] = T
            dt = T - t[j-1]
            dW = std_n[j-1] * np.sqrt(dt)
            Y_next = explicit_step(Y[j-1], dt, dW)
            if Y_next < 0:
                backstop_usage[j] = 2
                Y[j] = backstop(Y[j-1], dt, dW)
            else:
                Y[j] = Y_next
            break 
        if h < h_min:
            t[j] = t[j-1] + h_min
            dt = h_min
            dW = std_n[j-1] * np.sqrt(dt)
            Y[j] = backstop(Y[j-1], dt, dW)
        else:
            t[j] = t[j-1] + h
            dt = h
            dW = std_n[j-1] * np.sqrt(dt)
            Y_next = explicit_step(Y[j-1], dt, dW)
            if Y_next < 0:
                backstop_usage[j] = 2
                Y[j] = backstop(Y[j-1], dt, dW)
            else:
                Y[j] = Y_next  
    X = Y**2
    return t[:j+1], X[:j+1], backstop_usage[:j+1]


@njit
def adaptive_heston(k, lamda, theta, X_0, S_0, rf, cor, T, h_max, h_min, r):
    alpha = (4*k*lamda - theta**2)/8
    beta = -k/2
    gamma = theta/2
    
    max_steps = int(T//h_min + 1)
    t = np.zeros(max_steps)
    Y = np.zeros(max_steps)
    dWS = np.zeros(max_steps)
    
    backstop_usage = np.zeros(max_steps)
    
    Y[0] = np.sqrt(X_0)
    Y_temp = Y[0]
    
    for j in range(1, max_steps):
        h = h_max * np.minimum(1, np.abs(Y_temp)**r)
        t_next = t[j-1] + h
        
        if h < h_min and t_next < T:
            backstop_usage[j] = 1
            t[j] = t[j-1] + h_min
            W_inc = np.random.normal(0, np.sqrt(h_min))
            dWS[j-1] = cor*W_inc + np.random.normal(0, np.sqrt(h_min)) * np.sqrt(1-cor**2)
            discriminant = (Y_temp + gamma*W_inc)**2/(4*(1-beta*h)**2) \
                    + (alpha*h)/(1-beta*h)
            if discriminant < 0:
                Y[j] = 0
            else:
                Y_temp = (Y_temp + gamma*W_inc)/(2*(1-beta*h)) \
                         + np.sqrt(
                             np.abs(discriminant)
                         )
                Y[j] = Y_temp

        else:
            if t_next > T:
                h = T - t[j-1]
                t_next = T
            t[j] = t[j-1] + h
            W_inc = np.random.normal(0, np.sqrt(h))
            dWS[j-1] = cor*W_inc + np.random.normal(0, np.sqrt(h)) * np.sqrt(1-cor**2)
            Y_temp = Y_temp + h*(alpha/Y_temp + beta*Y_temp) + gamma * W_inc
            if Y_temp < 0:
                backstop_usage[j] = 2
                discriminant = (Y_temp + gamma*W_inc)**2/(4*(1-beta*h)**2) \
                    + (alpha*h)/(1-beta*h)
                if discriminant < 0:
                    Y[j] = 0
                else:
                    Y_temp = (Y_temp + gamma*W_inc)/(2*(1-beta*h)) \
                             + np.sqrt(
                                 np.abs(discriminant)
                             )
                    Y[j] = Y_temp
            else:
                Y[j] = Y_temp
        if t_next >= T:
            break

    X = Y**2
    t, X, backstop_usage = t[:j+1], X[:j+1], backstop_usage[:j+1]
    
    S = np.zeros(X.shape)

    S[0] = S_0
    S_temp = S[0]
    for j in range(1, X.shape[0]):
        W_inc = dWS[j-1]
        dt = t[j] - t[j-1]
        S_temp = S_temp + rf*S_temp*dt + np.sqrt(X[j-1])*S_temp*W_inc
        S[j] = S_temp
    return S
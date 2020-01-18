import numpy as np


def solve_em(f, g, X_0, t, W):
    X_em = np.zeros(W.shape)
    
    X_em[:,0] = X_0
    X_temp = np.copy(X_em[:,0])
    for j in range(1, W.shape[1]):
        W_inc = W[:,j] - W[:,j-1]
        dt = t[j] - t[j-1]
        X_temp = X_temp + dt*f(X_temp) + g(X_temp)*W_inc
        X_em[:,j] = X_temp
    return t, X_em


def implicit_scheme(k, lamda, theta, X_0, t, W):
    alpha = (4*k*lamda - theta**2)/8
    beta = -k/2
    gamma = theta/2

    Y_sol = np.zeros(W.shape)
    
    Y_sol[:,0] = np.sqrt(X_0)
    Y_temp = np.repeat(np.sqrt(X_0), W.shape[0])
    #set_trace()
    for j in range(1, W.shape[1]):
        W_inc = W[:,j] - W[:,j-1]
        dt = t[j] - t[j-1]
        discriminant = (Y_temp + gamma*W_inc)**2/(4*(1-beta*dt)**2) \
                       + (alpha*dt)/(1-beta*dt)

        Y_temp = (Y_temp + gamma*W_inc)/(2*(1-beta*dt)) \
                 + np.sqrt(
                     np.abs(discriminant)
                 )
        Y_temp[discriminant < 0] = 0
        Y_sol[:,j] = Y_temp
    X_sol = Y_sol**2
    return t, X_sol


def explicit_scheme(l, k, lamda, theta, X_0, t, W):
    X_sol = np.zeros(W.shape)
    X_sol[:,0] = np.sqrt(X_0)
    X_temp = np.copy(X_sol[:,0])
    
    for j in range(1, W.shape[1]):
        W_inc = W[:,j] - W[:,j-1]
        dt = t[j] - t[j-1]
        X_temp = ((1-dt*k/2)*np.sqrt(X_temp) + (theta*W_inc)/(2*(1-dt*k/2)))**2 + (lamda*k-theta**2/4)*dt + l*((W_inc)**2 - dt)
        X_temp[X_temp<0] = 0
        X_sol[:,j] = X_temp
    return t, X_sol
@njit
def single_adaptive(k, lamda, theta, X_0, T, h_max, h_min, r):
    alpha = (4*k*lamda - theta**2)/8
    beta = -k/2
    gamma = theta/2
    
    max_steps = int(T//h_min + 1)
    t = np.zeros(max_steps)
    Y = np.zeros(max_steps)
    std_n = np.random.randn(max_steps)
    
    backstop_usage = np.zeros(max_steps)
    # Code 2 for y next < 0
    # Code 1 for h < h_min
    
    Y[0] = np.sqrt(X_0)

    def backstop(Y, dt, dW):
        discriminant = (Y + gamma*dW)**2/(4*(1-beta*dt)**2) \
                    + (alpha*dt)/(1-beta*dt)
        if discriminant < 0:
            return 0
        else:
            return (Y + gamma*dW)/(2*(1-beta*dt)) \
                     + np.sqrt(
                         discriminant
                     )
    
    def explicit_step(Y, dt, dW):
        return Y + dt*(alpha/Y + beta*Y) + gamma * dW
    
    
    for j in range(1, max_steps):
        h = h_max * np.minimum(1, np.abs(Y[j-1])**r)
        if t[j-1] + np.maximum(h, h_min) > T: # If T exceeded
            t[j] = T # set final time to T
            dt = T - t[j-1] 
            dW = std_n[j-1] * np.sqrt(dt)
            Y_next = explicit_step(Y[j-1], dt, dW)
            if Y_next < 0: # If below zero
                backstop_usage[j] = 2 
                Y[j] = backstop(Y[j-1], dt, dW)
            else:
                Y[j] = Y_next
            break # break because T exceeded
        if h < h_min:
            t[j] = t[j-1] + h_min
            dt = h_min
            dW = std_n[j-1] * np.sqrt(dt)
            Y[j] = backstop(Y[j-1], dt, dW)
        else: # 'normal step'
            t[j] = t[j-1] + h
            dt = h
            dW = std_n[j-1] * np.sqrt(dt)
            Y_next = explicit_step(Y[j-1], dt, dW)
            # take explicit step. if under zero, use backstop
            if Y_next < 0:
                backstop_usage[j] = 2
                Y[j] = backstop(Y[j-1], dt, dW)
            else:
                Y[j] = Y_next  
    X = Y**2
    return t[:j+1], X[:j+1], std_n[:j] * np.sqrt(np.diff(t[:j+1])), backstop_usage[:j+1]


@njit
def single_heston_milstein(S_0, rf, cor, t_cir, X_cir, dW_V):
    S = np.zeros(X_cir.shape)
    dW_S = dW_V * cor + np.random.randn(len(dW_V)) * np.sqrt(np.diff(t_cir)) * np.sqrt(1-cor**2)
    #print(np.corrcoef(dW_V, dW_S))
    S[0] = S_0
    S_temp = S[0]
    for j in range(1, X_cir.shape[0]):
        dt = t_cir[j] - t_cir[j-1]
        #S_temp = S_temp + rf*S_temp*dt + np.sqrt(X[j-1])*S_temp*W_inc
        S_temp = S_temp + rf*S_temp*dt + np.sqrt(X_cir[j-1])*S_temp*dW_S[j-1] + \
            0.5*np.sqrt(X_cir[j-1])*S_temp * np.sqrt(X_cir[j-1])*(dW_S[j-1]**2 - dt)
        S[j] = S_temp
    return t_cir, S


@njit
def single_heston_log(S_0, rf, cor, t_cir, X_cir, dW_V):
    S = np.zeros(X_cir.shape)
    
    dW_S = dW_V * cor + np.random.randn(len(dW_V)) * np.sqrt(np.diff(t_cir)) * np.sqrt(1-cor**2)
    #print(np.corrcoef(dW_V, dW_S))
    S[0] = np.log(S_0)
    S_temp = S[0]
    for j in range(1, X_cir.shape[0]):
        dt = t_cir[j] - t_cir[j-1]
        #S_temp = S_temp + rf*S_temp*dt + np.sqrt(X[j-1])*S_temp*W_inc
        S_temp = S_temp + (rf - 0.5*X_cir[j-1])*dt + np.sqrt(X_cir[j-1]) * dW_S[j-1]
        S[j] = S_temp
    return t_cir, np.exp(S)


@njit
def heston_final_time(S_scheme, k, lamda, theta, X_0, T, h_max, h_min, r, S_0, rf, cor, M):
    S_final = np.zeros(M)
    X_final = np.zeros(M)
    
    h_mean_total = 0
    for i in range(M):
        t_cir, X_cir, dW_V, bs = single_adaptive(k, lamda, theta, X_0, T, h_max, h_min, r)
        h_mean_total += np.mean(np.diff(t_cir))
        X_final[i] = X_cir[-1]
        _, S = S_scheme(S_0, rf, cor, t_cir, X_cir, dW_V)
        S_final[i] = S[-1]
    
    return S_final, X_final, h_mean_total/M
    #return t_cir, X_cir, dW_V


@njit
def heston_milstein(S_0, r, t, W_S, CIR):
    X_em = np.zeros(W_S.shape)
    X_em[0] = S_0
    X_temp = X_em[0]
    for j in range(1, W_S.shape[0]):
        W_inc = W_S[j] - W_S[j-1]
        dt = t[j] - t[j-1]
        X_temp = X_temp + r*X_temp*dt + np.sqrt(CIR[j-1])*X_temp*W_inc \
        + 0.5*np.sqrt(CIR[j-1])*X_temp * np.sqrt(CIR[j-1])*(W_inc[j-1]**2 - dt)
        
        X_em[j] = X_temp
    return X_em


@njit
def heston_log(S_0, r, t, W_S, CIR):
    CIR = np.abs(CIR)
    X_em = np.zeros(W_S.shape)
    X_em[0] = np.log(S_0)
    X_temp = X_em[0]
    for j in range(1, W_S.shape[0]):
        W_inc = W_S[j] - W_S[j-1]
        dt = t[j] - t[j-1]
        X_temp = X_temp + (r - 0.5 * CIR[j-1])*dt + np.sqrt(CIR[j-1])*W_inc
        X_em[j] = X_temp
    return np.exp(X_em)


@njit
def fs_heston_final_time(scheme, S_scheme, k, lamda, theta, X_0, T, S_0, rf, cor, N, M, batches):
    X_T = np.zeros(M*batches)
    for i in range(batches):
        t, W1, W2 = correlated_paths(T, N, M, cor=cor)
        t, CIR = scheme(k, lamda, theta, X_0, t, W1)
        X_T[i*M:(i+1)*M] = S_scheme(S_0, rf, t, W2, CIR)[-1]
    return X_T
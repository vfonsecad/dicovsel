import numpy as np
import scipy as sp
from scipy.linalg import eigh




def covsel(X,Y0,nvar, scaleY = False, weights = None):
    
    Y = Y0.copy()
    
    if Y.ndim == 1:
        Y.shape = (Y.shape[0], 1)
    
    x_mu = np.mean(X, axis=0)
    y_mu = np.mean(Y, axis=0)


    n = X.shape[0]
    k = X.shape[1]
    ky = Y.shape[1]


    # centering x
    X_c = X - x_mu
    Y_c = Y - y_mu
    
    if scaleY:
        Y_c /= Y_c.std(axis=0)

    # ------ 1. covsel

    # total var
    
    if weights is None:
        
        weights = np.ones(n)/n
        
    weights_m = np.diag(weights)
    xsstot = np.sum(np.dot(np.dot(X_c.T, weights_m), X_c))
    ysstot = np.sum(np.dot(np.dot(Y_c.T, weights_m), Y_c))


    yss = np.zeros(nvar)
    xss = np.zeros(nvar)
    selvar = np.zeros(nvar).astype(np.int32)
    notselvar = np.zeros(k) == 0
    allvars = np.arange(k)
    


    for ii in range(nvar):

        Sxy = np.dot(np.dot(X_c.T, weights_m), Y_c)
        z = np.diag(np.dot(Sxy,Sxy.T))[notselvar]
        selvar[ii] = allvars[notselvar][np.argmax(z)]
        notselvar[selvar[ii]] = False
        u = X_c[:,selvar[ii]]
        u.shape = (u.shape[0], 1)
        u_inv = 1/np.dot(np.dot(u.T, weights_m), u)
        P = np.dot(np.dot(np.dot(u,u_inv), u.T), weights_m) # P = u %*% solve(t(u) %*% D %*% u) %*% t(u) %*% D
        X_c = X_c - np.dot(P,X_c)
        Y_c = Y_c - np.dot(P,Y_c) 

        xss[ii] = np.sum(np.dot(np.dot(X_c.T, weights_m), X_c))
        yss[ii] = np.sum(np.dot(np.dot(Y_c.T, weights_m), Y_c))
    
        
    selvar_final = selvar.copy()#np.sort(selvar)    
    cumpvarx = 1 - (xss/xsstot)
    cumpvary = 1 - (yss/ysstot)


    return (selvar,cumpvarx,cumpvary)
    
    

def convex_relaxation(xs_c, xt_c):
        '''
        Convex relaxation of covariance difference.
         
        The convex relaxation computes the eigenvalue decomposition of the (symetric) covariance 
        difference matrix, inverts the signs of the negative eigenvalues and reconstructs it again. 
        It can be shown that this relaxation corresponds to an upper bound on the covariance difference
        btw. source and target domain (see ref.)
        
        Reference:
        * Ramin Nikzad-Langerodi, Werner Zellinger, Susanne Saminger-Platz and Bernhard Moser 
          "Domain-Invariant Regression under Beer-Lambert's Law" In Proc. International Conference
          on Machine Learning and Applications, Boca Raton FL 2019.
        
        Parameters
        ----------
        
        xs_c: numpy array (Ns x k) centered
            Source domain matrix
            
        xt_c: numpy array (Nt x k) centered
            Target domain matrix
            
        Returns
        -------
        
        D: numpy array (k x k)
            Covariance difference matrix
        
        '''

        
        xs = xs_c.copy()
        xt = xt_c.copy()

        
        # Preliminaries
        ns = np.shape(xs)[0]
        nt = np.shape(xt)[0]
        
        # Compute difference between source and target covariance matrices   
        rot = (1/ns*xs.T@xs- 1/nt*xt.T@xt) 

        # Convex Relaxation
        w,v = eigh(rot)
        eigs = np.abs(w)
        eigs = np.diag(eigs)
        D = v@eigs@v.T 

        return D
    
    
def dicovsel(X,Y,Xs, Xt,nvar, l,scaleY = False, weights = None):
    
    if Y.ndim == 1:
        Y.shape = (Y.shape[0], 1)
    
    x_mu = np.mean(X, axis=0)
    y_mu = np.mean(Y, axis=0)
    
    xs_mu = np.mean(Xs, axis=0)
    xt_mu = np.mean(Xt, axis=0)
    


    n = X.shape[0]
    k = X.shape[1]
    ky = Y.shape[1]


    # centering x
    X_c = X - x_mu
    Y_c = Y - y_mu
    
    Xs_c = Xs - xs_mu
    Xt_c = Xt - xt_mu
    
    
    if scaleY:
        Y_c /= Y_c.std(axis=0)

    # ------ 1. covsel

    # total var
    
    if weights is None:
        
        weights = np.ones(n)
        
    weights_m = np.diag(weights)
    xsstot = np.sum(np.dot(np.dot(X_c.T, weights_m), X_c))
    ysstot = np.sum(np.dot(np.dot(Y_c.T, weights_m), Y_c))


    yss = np.zeros(nvar)
    xss = np.zeros(nvar)
    selvar = np.zeros(nvar).astype(np.int32)
    notselvar = np.zeros(k) == 0
    allvars = np.arange(k)
    z_signal = np.zeros((k, nvar))
    


    for ii in range(nvar):

        Sxy = np.dot(np.dot(X_c.T, weights_m), Y_c)
        D = convex_relaxation(Xs_c, Xt_c)
        lA = (Y_c.T.dot(Y_c))[0,0]*np.eye(D.shape[0]) + l*D
        z0 = sp.linalg.solve(lA.T, Sxy, assume_a='sym')  # ~10 times faster
        z_all = np.diag(np.dot(z0,z0.T))
        z_signal[:,ii] = z0[:,0] 
        z = z_all[notselvar]     
        selvar[ii] = allvars[notselvar][np.argmax(z)]
        notselvar[selvar[ii]] = False
        u = X_c[:,selvar[ii]]
        u.shape = (u.shape[0], 1)
        u_inv = 1/np.dot(np.dot(u.T, weights_m), u)
        P = np.dot(np.dot(np.dot(u,u_inv), u.T), weights_m) # P = u %*% solve(t(u) %*% D %*% u) %*% t(u) %*% D
        X_c = X_c - np.dot(P,X_c)
        Y_c = Y_c - np.dot(P,Y_c) 
        
        us = Xs_c[:,selvar[ii]]
        us.shape = (us.shape[0], 1)
        Ps = us.dot(1/(us.T.dot(us))).dot(us.T)
        Xs_c = Xs_c - np.dot(Ps,Xs_c)

        ut = Xt_c[:,selvar[ii]]
        ut.shape = (ut.shape[0], 1)
        Pt = ut.dot(1/(ut.T.dot(ut))).dot(ut.T)
        Xt_c = Xt_c - np.dot(Pt,Xt_c)
        
        
        
        

        xss[ii] = np.sum(np.dot(np.dot(X_c.T, weights_m), X_c))
        yss[ii] = np.sum(np.dot(np.dot(Y_c.T, weights_m), Y_c))
    
        
    selvar_final = selvar.copy()#np.sort(selvar)    
    cumpvarx = 1 - (xss/xsstot)
    cumpvary = 1 - (yss/ysstot)


    return (selvar,cumpvarx,cumpvary,z_signal)


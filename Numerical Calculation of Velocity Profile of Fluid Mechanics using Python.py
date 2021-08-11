import numpy as np
import matplotlib.pyplot as plt


h = 0.002 #the mesh size here is taken to 0.002 
k = (1/h) + 1 #k is  number of  discrete points in the approximation 
k = int(k) # to make sure k only belongs to an integer
eta_infinity = 3.5 # first guess of eta_infinity
eta_infinity_previous = 0 # second guess of eta_infinity for the secant method


f = np.array([0.0]*(k-1)) #taking the values of f in an array
for i in range(len(f)):
    f[i] = 0.5*(eta_infinity*(i+1)*h)#we have taken the initial value of f as given in the project paper. Page no. 388
f = f.reshape(500,1)



def create_J(eta_infinity,f,k,h): #Defining Jacobian as J. Entering the values of the matrix. The first three rows and the last two rows are entered manually, while a for loop is used for other entires. (General case)
    J = np.zeros((k-1,k-1))
    J[0,0] = 4/h
    J[0,1] = -3/h
    J[0,2] = 4/(3*h)
    J[0,3] = -1/(4*h)
    J[1,0] = 29/(h**3)
    J[1,1] = -461/(8*(h**3))
    J[1,2] = 62/(h**3)
    J[1,3] = -307/(8*(h**3))
    J[1,4] = 13/(h**3)
    J[1,5] = -15/(8*(h**3))
    J[2,0] = (-1/(4*(h**3))) + ((4*eta_infinity*f[3])/(3*(h**2)))
    J[2,1] = (5/(2*(h**3))) - ((5*eta_infinity*f[3])/(2*(h**2))) + (eta_infinity*(-f[1] + 16*f[2] - 30*f[3] +16*f[4] - f[5]))/(12*(h**2))
    J[2,2] = (-7/(2*(h**3))) + ((4*eta_infinity*f[3])/(3*(h**2)))
    J[2,3] = (7/(4*(h**3))) - ((eta_infinity*f[3])/(12*(h**2)))
    J[2,4] = (-1/(4*(h**3)))
    for p in range(3,k-3):
        i = p-1
        J[p,p-3] = (-1/(4*(h**3))) - ((eta_infinity*f[i])/(12*(h**2)))
        J[p,p-2] = (-1/(4*(h**3))) + ((4*eta_infinity*f[i])/(3*(h**2)))
        J[p,p-1] = (5/(2*(h**3))) - ((5*eta_infinity*f[i])/(2*(h**2))) + (eta_infinity*(-f[i-2] + 16*f[i-1] - 30*f[i] +16*f[i+1] - f[i+2]))/(12*(h**2))
        J[p,p] = (-7/(2*(h**3))) + ((4*eta_infinity*f[i])/(3*(h**2)))
        J[p,p+1] = (7/(4*(h**3))) - ((eta_infinity*f[i])/(12*(h**2)))
        J[p,p+2] = (-1/(4*(h**3)))
    J[k-3,k-8] = 15/(8*(h**3))
    J[k-3,k-7] = -13/(h**3)
    J[k-3,k-6] = 307/(8*(h**3))
    J[k-3,k-5] = -62/(h**3)
    J[k-3,k-4] = 461/(8*(h**3))
    J[k-3,k-3] = -29/(h**3)
    J[k-3,k-2] = 49/(8*(h**3))
    J[k-2,k-6] = 1/(4*h)
    J[k-2,k-5] = -4/(3*h)
    J[k-2,k-4] = 3/h
    J[k-2,k-3] = -4/h
    J[k-2,k-2] = 25/(12*h)
    return J


def create_y(eta_infinity,f,k,h): #Creating the Y vector and entering the values.
    y = np.zeros((k-1,1))
    k = k-2
    y[0] = (48*f[0]-36*f[1]+16*f[2]-3*f[3])/(12*h)
    y[1] = (232*f[0]-461*f[1]+496*f[2]-307*f[3]+104*f[4]-15*f[5])/(8*(h**3))
    for p in range(2,k-3):
        i = p - 1
        y[p] = (-f[i-2]-f[i-1]+10*f[i]-14*f[i+1]+7*f[i+2]-f[i+3])/(4*(h**3)) + (eta_infinity)*f[i]*(-f[i-2]+16*f[i-1]-30*f[i]+16*f[i+1]-f[i+2])/(12*(h**2))
    y[k-3] = (15*f[k-6]-104*f[k-5]+307*f[k-4]-496*f[k-3]+461*f[k-2]-232*f[k-1]+49*f[k])/(8*(h**3))
    y[k-2] = (3*f[k-4]-16*f[k-3]+36*f[k-2]-48*f[k-1]+25*f[k])/(12*h) - (eta_infinity)
    return y

error_delf = 1e-10
error_Z = 1e-15
Z = 10 #dummy value
while(abs(Z) > error_Z): #the outer while loop starts here for updating eta_infinity
    del_f = np.ones((k-1,1))
    while(np.linalg.norm(del_f,ord = np.inf)>=error_delf): #the inner while loop starts here for updating del_f
        J = create_J(eta_infinity_previous,f,k,h)
        y = create_y(eta_infinity_previous,f,k,h)
        del_f = np.linalg.solve(J,-y)
        print (max(del_f))
        f_prev = f
        f = f + del_f
    Z_prev = (35*f_prev[k-2]-104*f_prev[k-3]+114*f_prev[k-4]-56*f_prev[k-5]+11*f_prev[k-6])/(12*(h**2))
    Z = (35*f[k-2]-104*f[k-3]+114*f[k-4]-56*f[k-5]+11*f[k-6])/(12*(h**2))
    eta_infinity_previous = eta_infinity
    eta_infinity = eta_infinity - ((eta_infinity - eta_infinity_previous)/(Z - Z_prev))*Z


print (f)
print (eta_infinity)

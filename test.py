import numpy as np
from pyDOE import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pymc3 as pm
from matplotlib import cm
#x为n行2列的数据
def f_z(x,n):
    e=np.random.normal(loc=0,scale=0.1,size=n)
    # print(e)
    return 1.04+3.02*x[:,0]**3+2.01*x[:,1]**3+np.sin(x[:,0]*x[:,1]/4)+e

def f_y(x,theta,n):
    return 1+3*x[:,0]**3+2*x[:,1]**3+theta*x[:,0]*x[:,1]

def gen_z(n=10):
    x=lhs(2,n)
    return x,f_z(x,n)

def gen_y(theta=0.5,n=40):
    x=lhs(2,n)
    return x,f_y(x,theta,n)

def draw():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x,z=gen_z(n=100)
    ax.scatter(x[:,0],x[:,1],z,c='r',label='observation data')
    x,y=gen_y(n=100)
    ax.scatter(x[:,0],x[:,1],y,c='b',label='model data')
    ax.set_xlabel('X0 Label')
    ax.set_ylabel('X1 Label')
    ax.set_zlabel('Z Label')
    ax.legend()
    plt.show()

def stimulation():
    N=100
    model=pm.Model()
    with model:

        x,z=gen_z(n=100)
        
        alpha = 0.1
        ls = [0.2,0.2]
        tau = 2.0
        cov = tau * pm.gp.cov.RatQuad(2, ls=ls, alpha=alpha)
        Lambda=pm.Gamma('Lambda',alpha=2,beta=0.5)
        sigma = pm.Normal("sigma",mu=0,sd=Lambda)
        gp = pm.gp.Marginal(cov_func=cov)

        y_ = gp.marginal_likelihood("y", X=x, y=z, noise=sigma) 
        Xnew,geny=gen_y(n=N)
        y_star = gp.conditional("y_star", Xnew=Xnew, pred_noise=True)

        mp = pm.find_MAP()

    print(mp)
    print(mp['y_star'].shape)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # X = np.linspace(0, 1, 10)
    # Y = np.linspace(0, 1, 10)
    # X, Y = np.meshgrid(X, Y)
    # Z= 1.04+3.02*X**3+2.01*Y**3+np.sin(X*Y/4)+np.random.normal(loc=0,scale=0.1,size=1)
    # ax.plot_surface(X, Y, Z,alpha=0.5)
    
    x=Xnew
    ydata=mp['y_star']
    ax.scatter(x[:,0],x[:,1],ydata,c='r')

    ax.set_xlabel('X0 Label')
    ax.set_ylabel('X1 Label')
    ax.set_zlabel('Z Label')
    ax.set_title(['blue is real data,red is predict data'])
    plt.show()

if __name__=='__main__':
    stimulation()  


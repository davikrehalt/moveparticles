#simulating motion
import numpy as np
import math
from math import sqrt,pi,sin,cos
import matplotlib.pyplot as plt
import copy
import pygame
from pygame.locals import *
from mpl_toolkits.mplot3d import Axes3D

def polygon0(cen,k,r,n,d=[1.0,1.0]):
    #adds n movable particles in a polygon
    cen=np.array(cen) #position of center
    k=np.array(k) #rotation direction
    r=np.array(r) #displacement of the first point
    lis=[]
    for i in range(n):
        theta=i*2*pi/n
        p=r*cos(theta)+np.cross(k,r)*sin(theta)+k*np.dot(k,r)*(1-cos(theta)) #rodrigues' rotation
        lis.append([list(p+cen),d,[0.0,0.0,0.0]])
    return lis

def polygon1(cen,k,r,n,d=1.0):
    #adds n stationary particles in a polygon
    cen=np.array(cen) #position of center
    k=np.array(k) #rotation direction
    r=np.array(r) #displacement of the first point
    lis=[]
    for i in range(n):
        theta=i*2*pi/n
        p=r*cos(theta)+np.cross(k,r)*sin(theta)+k*np.dot(k,r)*(1-cos(theta)) #rodrigues' rotation
        lis.append([list(p+cen),d])
    return lis
                
    
control=1 #simulate
iterations=1000
sw=1 #switch display of pygame

#fixed particles
Lc=[([-2.0,0.0,0.0],-1.0),([2.0,0.0,0.0],1.0)] #the position and charge
Lm=[] #the position and mass

#free particles
l=[[[-0.3,0.3,0.0],[1.0,1.0],[0.0,0.5,0.0]]] #free particles pos, q,m , vel

tot=len(l)
#f1=899180 #actual k_e
#f2=6.6738*10**(-11) #actual G

f1=1.0 #f/(q1q2/r^2) #factor of electrostatics
if f1!=0:
    epsil=1.0/(4*pi*f1)
f2=-0.0 #f/(m1m2/r^2) #factor of gravity (negative)
f3=0.0 #magnetism
mu=f3*4*pi

O=np.zeros(3) #origin

gre=np.array([0.0,0.0,0.0]) #background electric field
grg=np.array([0.0,0.0,0.0]) #background gravity
grb=np.array([0.0,0.0,0.0]) #background magnetic

def dist(x):
    return sqrt(1.0*np.dot(x,x))
    
def Center(lt=l):
    #COM
    lis=[]
    ml=[]
    for i,j in enumerate(lt):
        m=j[1][1]
        lis.append(m*np.array(j[0]))
        ml.append(m)
    mt=np.sum(ml)
    return np.sum(lis,0)*1.0/mt
    
def Efield(x):
    #the electric field without free particles
    #k*sum(q/r^2 *r_hat)    
    x=np.array(x)
    if len(Lc) == 0:
        return np.zeros(3)+gre
    return f1*np.sum([(i[1]*1.0*(x-i[0])/dist(x-i[0])**3) for i in Lc],0)+gre
    
def Gfield(x):
    #gravitational field without free
    #k*sum(m/r^2 *r_hat)
    x=np.array(x)
    if len(Lm) == 0:
        return np.zeros(3)+grg
    return f2*np.sum([(i[1]*1.0*(x-i[0])/dist(x-i[0])**3) for i in Lm],0)+grg
    
def Bfield(x):
    #magnetic field without free
    return grb
    
def pot1(x):
    #electric potential b/c of fixed points
    x=np.array(x)
    bck=-1.0*np.dot(gre,x) #background pot V=-E.x (dot product)
    return f1*sum([(i[1]/dist(x-i[0])) for i in Lc])+bck
    
def pot2(x):
    #grav pot b/c of fixed points
    x=np.array(x)
    bck=-1.0*np.dot(grg,x) #background pot 
    return f1*sum([(i[1]/dist(x-i[0])) for i in Lm])+bck

def Etot(x):
    #total electric field
    x=np.array(x)
    El=[f1*j[1][0]*(x-np.array(j[0]))/dist(x-np.array(j[0]))**3 for j in l] #sum of free elec. 
    return np.sum(El,0)+Efield(x)
    
def Gtot(x):
    #total grav. field
    x=np.array(x)
    Gl=[f2*j[1][1]*(x-np.array(j[0]))/dist(x-np.array(j[0]))**3 for j in l] #the sum of free grav effects
    return np.sum(Gl,0)+Gfield(x)
    
def Btot(x):
    #magnetic field
    x=np.array(x)
    Bl=[0.0 for i in range(tot)]
    for i,j in enumerate(l):
        Bl[i]=f3*j[1][0]*np.cross(j[2],x-np.array(j[0]))/dist(x-np.array(j[0]))**3
    return np.sum(Bl,0)+Bfield(x)
    
def force(n,lt=l,show=[]):
    #the force of the ith particle
    pos=np.copy(np.array(lt[n][0]))
    [q,m]=lt[n][1][:]
    v=np.copy(lt[n][2])
    f=[]
    for i,j in enumerate(lt):
        if i!=n:
            d=pos-np.array(j[0])
            fe=f1*q*j[1][0]*d/dist(d)**3 #electric
            fg=f2*m*j[1][1]*d/dist(d)**3 #grav
            fb=f3*q*j[1][0]*np.cross(v,np.cross(j[2],d))/dist(d)**3 #lorentz magnetic
            if show == 1:
                print fe
                print fg
                print fb
            f.append(fe+fg+fb)
    if show == 1:
        print (Efield(pos)+np.cross(v,Bfield(pos)))*q
        print Gfield(pos)*m
        print np.sum(f,0)
    return (Efield(pos)+np.cross(v,Bfield(pos)))*q+Gfield(pos)*m+np.sum(f,0)

def V(lt=l):
    #potential total
    ret=[]
    for i,j in enumerate(lt):
        lis=[]
        q=j[1][0]
        m=j[1][1]
        for a,b in enumerate(lt[i+1:]):
            d=np.array(j[0])-np.array(b[0])
            lis.append(f1*q*b[1][0]/dist(d)+f2*m*b[1][1]/dist(d)) #all mutual potential
        ret.append(np.sum(lis)+pot1(j[0])*q+pot2(j[0])*m) #potential to stationary
    return sum(ret)
    
def T(lt=l):
    #kinetic
    return sum([0.5*j[1][1]*np.dot(np.array(j[2]),np.array(j[2])) for j in lt])

def lmom(lt=l):
    #linear momentum
    return np.sum([j[1][1]*np.array(j[2]) for j in lt],0)

def amom(lt=l):
    #angular momentum
    return np.sum([np.cross(j[1][1]*np.array(j[2]),np.array(j[0])) for j in lt],0)
def E():
    return T()+V()
def pos(n,lt=l):
    return np.copy(lt[n][0])
def vel(n,lt=l):
    return np.copy(lt[n][2])
def acc(n,lt=l):
    return np.copy(force(n,lt)/lt[n][1][1])

def step1(dt):
    #euler
    H0=T()+V() #energy
    x0=[0.0 for i in range(tot)]
    v0=[0.0 for i in range(tot)]
    a0=[0.0 for i in range(tot)]
    dx=[0.0 for i in range(tot)]
    dv=[0.0 for i in range(tot)]
    da=[0.0 for i in range(tot)]
    for n in range(tot):  
        x0[n]=pos(n)
        v0[n]=vel(n)
        a0[n]=acc(n)
        dx[n]=v0[n]*dt+0.5*a0[n]*dt**2 #displacement
    for n in range(tot):
        l[n][0]=x0[n]+dx[n] #update position
    for n in range(tot):
        da[n]=acc(n)-a0[n] #acc diff
        dv[n]=(a0[n]+0.5*da[n])*dt #velocity diff
        l[n][2]=v0[n]+dv[n] #velocity update
        
    H1=T()+V()
    err=sum([dist(da[n]*dt**2*0.5)**2 for n in range(tot)]) #max error in position
    er2=H1-H0 #error in energy
    return err,er2

def step2(dt):
    #runge-kutta 4
    H0=T()+V() #energy
    x0=[0.0 for i in range(tot)]
    v0=[0.0 for i in range(tot)]
    a0=[0.0 for i in range(tot)]
    dx=[0.0 for i in range(tot)]
    dv=[0.0 for i in range(tot)]
    da=[0.0 for i in range(tot)]
    k=[[0.0 for i in range(tot)] for i in range(4)] #dx terms of v
    j=[[0.0 for i in range(tot)] for i in range(4)] #dv terms of a
    l1=copy.deepcopy(l)
    l2=copy.deepcopy(l)
    l3=copy.deepcopy(l)
    for n in range(tot):  
        x0[n]=pos(n)
        v0[n]=vel(n)
        a0[n]=acc(n)
        k[0][n]=a0[n] #the first terms
        j[0][n]=v0[n]
        l1[n][0]=x0[n]+0.5*j[0][n]*dt #the first change to mid-intervel for the second terms
        l1[n][2]=v0[n]+0.5*k[0][n]*dt
    for n in range(tot):
        k[1][n]=acc(n,l1) #second terms
        j[1][n]=vel(n,l1)
        l2[n][0]=x0[n]+0.5*j[1][n]*dt #for the third terms
        l2[n][2]=v0[n]+0.5*k[1][n]*dt
    for n in range(tot):
        k[2][n]=acc(n,l2) #third terms
        j[2][n]=vel(n,l2)
        l3[n][0]=x0[n]+j[2][n]*dt #for the fourth terms
        l3[n][2]=v0[n]+k[2][n]*dt
    for n in range(tot):
        k[3][n]=acc(n,l3)
        j[3][n]=vel(n,l3)
        dv[n]=1.0/6.0*dt*(k[0][n]+2*k[1][n]+2*k[2][n]+k[3][n])
        dx[n]=1.0/6.0*dt*(j[0][n]+2*j[1][n]+2*j[2][n]+j[3][n])
        l[n][0]=x0[n]+dx[n]
        l[n][2]=v0[n]+dv[n]
    for n in range(tot):
        da[n]=acc(n)-a0[n] #acc diff
    H1=T()+V()
    err=sum([dist(da[n]*dt**2*0.5)**2 for n in range(tot)]) #max error in position
    er2=H1-H0 #error in energy
    return err,er2

def space(timel,vectl,interval):
    #evenly space the data points
    a=-int(-1.0*timel[0]/interval)
    b=int(timel[-1]/interval)
    stl=np.arange(a*interval,(b)*interval,interval)
    ret=[]
    k=0
    for i,j in enumerate(timel):
        if j==stl[k]:
            ret.append(vectl[i])
            k+=1
            if k == len(stl):
                    break
        elif j<stl[k]:
            if timel[i+1]>stl[k]:
                r0=stl[k]-j
                r1=timel[i+1]-stl[k]
                ret.append((r1*vectl[i]+r0*vectl[i+1])/(r1+r0)) #interpolate
                k+=1
                if k == len(stl):
                    break
        else:
            return False

    return ret

def space2(timel,posl,interval):
    #evenly space the data points
    a=-int(-1.0*timel[0]/interval)
    b=int(timel[-1]/interval)
    stl=np.arange(a*interval,(b)*interval,interval)
    ret=[]
    k=0
    for i,j in enumerate(timel):
        if abs(j-stl[k])<10**(-8):
            ret.append(posl[i])
            k+=1
            if k == len(stl):
                    break
        elif j<stl[k]:
            if i+1 == len(timel):
                print k
                print stl[k]
                print timel[-1]
                raise ValueError
            if timel[i+1]>stl[k]:
                r0=stl[k]-j
                r1=timel[i+1]-stl[k]
                ret.append([(r1*posl[i][m]+r0*posl[i+1][m])/(r1+r0) for m in range(len(posl[i]))]) #interpolate
                k+=1
                if k == len(stl):
                    break
        else:
            print j
            print stl[k]
            return False

    return ret

def pltp1(xmin,xmax,ymin,ymax,z=0,res=0.05,cap=20):
    #cap is a limit on z
    #plot potential
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    x=np.arange(1.0*xmin,1.0*xmax,res)
    y=np.arange(1.0*ymin,1.0*ymax,res)
    X, Y = np.meshgrid(x, y)
    zs = np.array([max(-cap,min(cap,pot1([x,y,z]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X,Y,Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    
def pltp2(xmin,xmax,ymin,ymax,z=0,res=0.05,cap=20):
    #cap is a limit on z
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    x=np.arange(1.0*xmin,1.0*xmax,res)
    y=np.arange(1.0*ymin,1.0*ymax,res)
    X, Y = np.meshgrid(x, y)
    zs = np.array([max(-cap,min(cap,pot2([x,y,z]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    ax.plot_surface(X,Y,Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plotpath(posl):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    for i in range(tot):
        ql=[j[i] for j in posl] #3d position
        xl=[j[0] for j in ql]
        yl=[j[1] for j in ql]
        zl=[j[2] for j in ql]
        ax.plot(xl,yl,zl)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    return True

def plot3d(lis):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    xl=[j[0] for j in lis]
    yl=[j[1] for j in lis]
    zl=[j[2] for j in lis]
    ax.plot(xl,yl,zl)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    return True

def show():
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    xl=[]
    yl=[]
    zl=[]
    for j in l:
        xl.append(j[0][0])
        yl.append(j[0][1])
        zl.append(j[0][2])
    ax.scatter(xl,yl,zl,c='r')
    for j in Lc:
        xl.append(j[0][0])
        yl.append(j[0][1])
        zl.append(j[0][2])
    ax.scatter(xl,yl,zl,c='k')
    for j in Lm:
        xl.append(j[0][0])
        yl.append(j[0][1])
        zl.append(j[0][2])
    ax.scatter(xl,yl,zl,c='b')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    return True
def plot2dt(tl,posl):
    #plot the 2d path over time
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    for i in range(tot):
        ql=[j[i] for j in posl] #3d position
        xl=[j[0] for j in ql]
        yl=[j[1] for j in ql]
        zl=[j[2] for j in ql]
        ax.plot(xl,yl,tl)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('t Label')
    plt.show()
    return True


#simulate
print E(),lmom(),amom()
t=0
ti=0.01 #original time increment
dt=ti
tl=[] #time list
pl=[] #position list
vl=[]
al=[]
el=[] #error for energy
coml=[] #center of mass
q=iterations #iteration
if control == 0:
    q=0 #off if control == 0
while True:
    if q == 0:
        break
    #new entry
    tl.append(t)
    pl.append([pos(n) for n in range(tot)])
    vl.append([vel(n) for n in range(tot)])
    al.append([acc(n) for n in range(tot)])
    coml.append(Center())
    #increment
    t+=dt
    (a,b)=step2(dt)
    #check errors
    el.append(b)
    if max(abs(a),abs(b))>0.0001:
        #max error
        dt=dt/3.0
        if max(abs(a),abs(b))>0.01:
            print 'error'
            q=0
    if max(abs(a),abs(b))<10**(-10):
        #too low an error
        #increase the time interval
        dt=min(ti,dt*2)
    #check if exit
    if len(tl)>=q:
        break
if q!=0:
    el=np.array(el)
    tl=np.array(tl)
    #reformulate the array
    xl=np.array([j[0][0] for j in pl])
    yl=np.array([j[0][1] for j in pl])
    zl=np.array([j[0][2] for j in pl])
    vxl=np.array([j[0][0] for j in vl])
    vyl=np.array([j[0][1] for j in vl])
    axl=np.array([j[0][0] for j in al])
    ayl=np.array([j[0][1] for j in al])
    #x1l=np.array([j[1][0] for j in pl])
    #y1l=np.array([j[1][1] for j in pl])
    #plt.plot(tl,xl)

    #plt.show()
    print 'simulated'



#game window
def draw(vl):
    #draws the position
    S.fill((0,0,0))
    for j in vl:
        pygame.draw.circle(S, c, (int(100*j[0])+400,-int(100*j[1])+300), 2)
    for j in Lc:
        if j[1]!=0:
            pygame.draw.circle(S, c, (int(100*j[0][0])+400,-int(100*j[0][1])+300), 2)
    for j in Lm:
        if j[1]!=0:
           pygame.draw.circle(S, c, (int(100*j[0][0])+400,-int(100*j[0][1])+300), 2)
    Surface.blit(S,(0,0))
    pygame.display.flip()
def GetInput():
    keystate = pygame.key.get_pressed()
    for event in pygame.event.get():
        if event.type == QUIT or keystate[K_ESCAPE]:
            pygame.quit();


if q == 0:
    sw=0 #switch

if sw == 1:
    #the display model of pygame
    
    wait=0.01 #wait 100 ms
    Dt=0.01 #move 100 ms
    leng=int(tl[-1]/Dt) #intervals
    
    #the evenly spaced
    Tl=np.arange(0,leng*Dt,Dt) #time
    Pl=space2(tl,pl,Dt) #position
    
    #pygame show
    pygame.init()
    c=(255,255,255)
    Surface = pygame.display.set_mode((800,600))
    S = pygame.Surface((800,600))
    
    i=0 #interval
    while True:
        GetInput()
        try:
            draw(Pl[i]) #ith point 
        except:
            break
        pygame.time.delay(int(wait*1000))
        i+=1
        if i == leng:
            pygame.quit();
            break

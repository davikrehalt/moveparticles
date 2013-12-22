#!/usr/bin/python
from math import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#motion of connected particles (connected with springs)

particlelist=[]
linklist=[]
externalforce=np.array([0.0,0.0,0.0])


#time interval
dt=0.1
N=5000 #iterations
f=0.00*dt #damping
stringconstant=50.0

def mag(a):
    return sqrt(np.dot(a,a))
def dist(a,b):
    return mag(b-a)

class particles():
    #particles in the link
    def __init__(self,mass,position,velocity,fixed=0):
        self.mass=mass
        self.pos=position
        self.vel=velocity
        self.force=np.zeros(3)
        self.fixed=fixed
    def getkinetic(self):
        return 0.5*self.mass*self.velocity**2
    def zeroforce(self):
        #zeros the force for new iteration
        self.force=np.zeros(3)+externalforce
        
class link():
    #the links
    def __init__(self,a,b,sconst,length):
        self.a=a
        self.b=b
        self.left=particlelist[a]
        self.right=particlelist[b]
        self.length=length #relaxed state
        self.sconst=sconst #string constant
    def getforce(self):
        displace=(self.right.pos-self.left.pos) #left to right vector
        delta=mag(displace)-self.length #from relaxed
        self.forceleft=delta*self.sconst*displace/mag(displace)
        self.forceright=-delta*self.sconst*displace/mag(displace)
    def potential(self):
        delta=dist(self.right.pos,self.left.pos)-self.length #from relaxed
        return 0.5*self.sconst*delta**2
    def push(self):
        self.left.force+=self.forceleft
        self.right.force+=self.forceright
    def update(self):
        self.getforce()
        self.push()

def addparticle(mass,position,ivelo=np.zeros(3),fixed=0):
    #adds particle
    particlelist.append(particles(mass*1.0,np.array(position)*1.0,np.array(ivelo)*1.0,fixed))
    return True

def addlink(a,b,sconst,length):
    #adds link
    linklist.append(link(a,b,sconst*1.0,length*1.0))

def makelinks(sconst):
    #makes links in order with original length
    n=len(particlelist)
    for j in range(n-1):
        l=dist(particlelist[j].pos,particlelist[j+1].pos)
        addlink(j,j+1,sconst,l)

    
def record():
    #record current state and update forces
    temp1=[]
    temp2=[]
    for particle in particlelist:
        particle.zeroforce()
        temp1.append(np.copy(particle.pos))
        temp2.append(np.copy(particle.vel))
    for link in linklist:
        link.update() #get all the forces
    return temp1,temp2
    
def updateall1(dt,n):
    #uses stormer's method
    positions=[[]]
    velocities=[[]] #list of pos,vel
    #0th iteration
    for particle in particlelist:
        particle.zeroforce()
        #initial recording
        positions[0].append(np.copy(particle.pos))
        velocities[0].append(np.copy(particle.vel))
        
    for link in linklist:
        link.update() #get all the forces
    for particle in particlelist:
        if particle.fixed == 0:
            #find the movement & move
            a=particle.force/particle.mass
            v=particle.vel
            particle.vel+=a*dt
            particle.pos+=0.5*a*dt**2+v*dt
            
    [newpos,newvel]=record() #record & update
    positions.append(newpos)
    velocities.append(newvel)
    
    for i in range(n):
        #iteration
        for num,particle in enumerate(particlelist):
            if particle.fixed == 0:
                #find the movement & move
                a=(particle.force/particle.mass)
                particle.pos+=(1-f)*positions[-1][num]-(1-f)*positions[-2][num]+a*dt**2
                particle.vel=(positions[-1][num]-positions[-2][num])/dt #not reliable--place filler
                
        [newpos,newvel]=record() #record&update
        positions.append(newpos)
        velocities.append(newvel)
        
    return positions,velocities

#setup
addparticle(1.0,[0,0,0],fixed=1)
addparticle(1.0,[1.0,0,0],[1,0,0])
addparticle(1.0,[2.0,0,0])
addparticle(1.0,[3.0,0,0])
addparticle(1.0,[4.0,0,0])
addparticle(1.0,[5.0,0,0])
addparticle(1.0,[6.0,0,0],fixed=1)
#links
makelinks(stringconstant)


#simulate
posts,velos=updateall1(dt,N)
#testing
zl=[k[1][2] for k in posts]
vzl=[k[1][2] for k in velos]
#make the figure
class demostrate():
    #plots the xz axis
    def __init__(self,sl):
        #sl side length
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, autoscale_on=False, xlim=(-sl, sl), ylim=(-sl, sl))
        self.ax.grid()

        #line&text template
        self.lines = [] #line template
        for link in linklist:
            lobj,= self.ax.plot([],[],'o-',lw=2,color='b')
            self.lines.append(lobj)
        self.time_template = 'time = %.1fs' #time template
        self.time_text = self.ax.text(0.05, 0.9, '', transform=self.ax.transAxes)
        self.run()
        
    def init(self):
        for line in self.lines:
            line.set_data([], [])
        self.time_text.set_text('')
        return list(self.lines)+[self.time_text]

    def animate(self,i):
        #ith iteration
        temppos=posts[i]
        for j,line in enumerate(self.lines):
            #plot each line segment
            line.set_data([temppos[linklist[j].a][0],temppos[linklist[j].b][0]],[temppos[linklist[j].a][2],temppos[linklist[j].b][2]])
            #line.set_data([0,2],[1,2])
        self.time_text.set_text(self.time_template%(i*dt))
        return list(self.lines)+[self.time_text]
    def run(self):
        ani = animation.FuncAnimation(self.fig, self.animate, np.arange(1,len(posts)),
        interval=25, blit=True, init_func=self.init)
        plt.show()


def plotposition(xl):
    fig=plt.figure()
    plt.plot(xl)
    plt.show()

def plotposition2(xl,yl):
    fig=plt.figure()
    plt.plot(xl,yl)
    plt.show()

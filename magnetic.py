from __future__ import absolute_import, division, print_function, unicode_literals

from datetime import time

import numpy as np
from numpy import random
import math
import matplotlib
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
from tensorflow import keras


matplotlib.use('TkAgg')

def dot(x,y): #contraction of arrays
    if max(x.ndim,y.ndim)>1:
        return np.sum(x*y,axis=1)
    else:
        return np.dot(x,y)
class Magnet:
    def __init__(self,xy,magnetisation=100,angle=0):
        self.strength=magnetisation
        self.position=xy
        self.angle=angle
        self.imperfect=False
        self.angular_velocity=0
        self.max_acceleration=40*2*3.14
    def set_imperfection(self,n=40,max=0.04): #small peturbations to dipole field
        self.imperfection_coefficients=(np.random.rand(4,n)-0.5)*max*self.strength
        self.imperfect=True
        self.n=n

    def imperfections(self,rel_angle): #unphysical peturbation, doesnt depend on distance
        angles=np.arange(0,self.n)*rel_angle[:,None]
        cosines=np.sum(self.imperfection_coefficients[0]*np.cos(angles),axis=1)
        sines=np.sum(self.imperfection_coefficients[1]*np.sin(angles),axis=1)
        cosines2 = np.sum(self.imperfection_coefficients[2] * np.cos(angles),axis=1)
        sines2 = np.sum(self.imperfection_coefficients[3] * np.sin(angles),axis=1)
        return np.stack((cosines+sines,cosines2+sines2),axis=1)


    def m(self): #magnetisation vector
        return self.strength*np.array([math.sin(self.angle),math.cos(self.angle)])

    def n(self): #normal vector
        return np.array([math.sin(self.angle), math.cos(self.angle)])

    def field(self,xy): #individual field

        if xy.ndim==1:
            xy.reshape((1,2))
            one_coord=True
        else:
            one_coord=False

        r=(xy-self.position)
        R=np.sqrt(dot(r,r))
        rel_angle=np.arctan2(r[:,0],r[:,1])-self.angle
        value=3*r*dot(self.m(),r)[:,None]/(R**5)[:,None]-(1/(R**3))[:,None]*self.m()
        if self.imperfect==True:
            value+=self.imperfections(rel_angle)

        if one_coord:
            value.reshape(2)

        return value
    def look_at(self,xy): #orient to point at xy
        r=(xy-self.position)
        new_angle=np.arctan2(r[0],r[1])
        self.angle=new_angle
    def probe(self,xy):
        self.look_at(xy)
        value=[]
        value2=[]
        angles = np.linspace(0, 6.28, 200)+self.angle
        for angle in angles:
            self.angle = angle
            value.append(self.field(xy.reshape((1, 2)))[0, 0])
            value2.append(self.field(xy.reshape(1, 2))[0, 1])
        plt.plot(value)
        plt.plot(value2)
        plt.show()

class Trajectory: #trajectory class, not used
    def __init__(self,time_max,coeff):
        self.time_max=time_max
        self.fn=coeff

    def randomize(self,max):
        self.fourier_series=self.n*max*(np.random.random((2,self.fn))-0.5)
        print(self.fourier_series)

    def FT(self):
        self.im_FT=np.fft.fft(self.value*1j+self.time)[0:self.fn]
        self.fourier_series=np.zeros((2,self.fn))
        self.fourier_series[0]=np.imag(self.im_FT)
        self.fourier_series[1]=-np.real(self.im_FT)
        print(self.fourier_series)

    def set_coords(self, value):
        self.value = value
        self.time=np.linspace(0, self.time_max, self.n)
        self.n=value.size
        self.FT()
    def plot_FT(self):
        plt.plot(np.fft.irfft(self.fourier_series[0]+1j*self.fourier_series[1],n=self.n))
        plt.show()
    def y(self):
        return np.fft.irfft(self.fourier_series[0]+1j*self.fourier_series[1],n=self.n)
    def normalised_FT(self):
        return self.fourier_series/self.max

def trajectory_fit(trajectory1, trajectory2):
    return np.sum((trajectory1.y-trajectory2.y)**2)/trajectory1.n











class Array: #class for full magnet array
    def __init__(self,XY=np.array([[1,1],[1,-1],[-1,1],[-1,-1]])):
        self.magnets=[]
        for xy in XY:
            self.magnets.append(Magnet(xy=xy))
            self.magnets[-1].set_imperfection(max=0.01)
        self.n=XY.shape[0]
    def field(self,XY): #sum of all fields
        UV=np.zeros(XY.shape)
        for magnet in self.magnets:
            UV+=magnet.field(XY)
        return UV
    def field_jacobian(self,xy): #Jacobian of field at xy w.r.t. magnet angles , would be good to replace with numpy implementation
        delta=1e-6
        jacobian=np.zeros((2,len(self.magnets)))
        for i in range(2):
            for j in range(len(self.magnets)):
                im_magnet=copy.deepcopy(self.magnets[j])
                im_magnet.angle+=delta
                jacobian[i,j]=(im_magnet.field(xy)[0,i]-self.magnets[j].field(xy)[0,i])/delta
        return jacobian


    def vector_plot(self,n=90,a=1.1): #plot overall field lines
        x = np.linspace(-a, a, n)
        y = np.linspace(-a, a, n)
        xx,yy=np.meshgrid(x, y)
        XY = np.stack((xx.flatten(),yy.flatten()),axis=1)
        UV=np.zeros(XY.shape)
        UV=self.field(XY)

        UV=UV/np.sqrt(dot(UV,UV))[:,None]*np.log(np.sqrt(dot(UV,UV)))[:,None] #log scale
        q = plt.quiver(XY[:,0], XY[:,1],UV[:,0] ,UV[:,1] ,scale=None, color='r')

        plt.show()
    def trajectory_field(self, fourier_coeff, xy=np.array([0, 0]), time=1,n=1000):

        self.Trajectories=[]
        for i in range(fourier_coeff.shape[0]):
            self.Trajectories.append(Trajectory(time_max=time,coeff=20))
            self.Trajectories[-1].fourier_series=fourier_coeff[i]
            self.Trajectories[-1].n=n

        angles=np.zeros((4,n))
        for i in range(angles.shape[0]):
            angles[i]=self.Trajectories[i].y()
        print(angles.shape)
        return self.probe(angles)
    def probe(self,angles,xy=np.array([0,0])): #plot field values at xy as angles are changing, needs rewriting
        value = []
        value2 = []
        for i in range(angles.shape[1]):
            for j in range(angles.shape[0]):
                self.magnets[j].angle = -angles[j,i]

            value.append(self.field(np.array([0,0]).reshape((1,2)))[0,0])
            value2.append(self.field(np.array([0,0]).reshape(1,2))[0,1])
        plt.plot(value)
        plt.plot(value2)
        plt.show()

        return np.array(value),np.array(value2)
    def orient(self,xy): #point all magnets to xy
        for magnet in self.magnets:
            magnet.look_at(xy)

def controller(magnet_array,target_trajectory,time_step,xy): #heuristic controller
    accelerations=np.array([0,0,0,0],dtype=np.double)
    n=target_trajectory.shape[0]
    velocities=np.array([0,0,0,0],dtype=np.double)
    angles=np.array([0,0,0,0],dtype=np.double)
    resultant_field=np.zeros((n,2))
    positions=np.zeros(n)
    for i in range(len(magnet_array.magnets)):
        angles[i]=magnet_array.magnets[i].angle

    print(magnet_array.field(xy))
    for i in range(0,n):
        accelerations=-dot((magnet_array.field(xy)[0]-target_trajectory[i]),magnet_array.field_jacobian(xy).transpose())
        velocities+=accelerations*time_step-velocities*5*time_step
        angles+=velocities*time_step
        for j in range(len(magnet_array.magnets)):
            magnet_array.magnets[j].angle=angles[j]

        resultant_field[i]=magnet_array.field(xy)[0]
        positions[i]=angles[0]

    print(magnet_array.field(xy)[:,0])
    plt.plot(resultant_field[:, 0])
    plt.plot(target_trajectory[:, 0])

    plt.plot(resultant_field[:,1])
    plt.plot(target_trajectory[:,1])
    plt.show()




model = keras.Sequential([
    keras.layers.Flatten(input_shape=(20, 2)),
    keras.layers.Dense(160, activation='relu'),
    keras.layers.Dense(160, activation='relu'),
    keras.layers.Reshape((4,2,20))
])
optimizer = tf.keras.optimizers.RMSprop(0.001)

coeffs=5000*(np.random.random((4,2,4))-0.5)
np.concatenate((coeffs,np.zeros((4,2,16))),axis=2)
magnets=Array()
magnets.orient(np.array([0,0]))
magnets.trajectory_field(coeffs)
print(magnets.field_jacobian(np.array([0,0])[None,:]))
graph=Magnet(np.array([-1,0]))
graph.set_imperfection()
graph.probe(np.array([0,0]))
n=10000
time=10

trajectoryx=Trajectory(value=np.concatenate((np.linspace(0,100,500),100-np.linspace(0,100,500)),axis=0),time_max=10,coeff=20)
trajectoryx.randomize(max=25)
#trajectoryx.FT()

trajectoryx.plot_FT()

#trajectoryx.plot_FT()
t=np.linspace(0,10,1000)
timestep=0.01
Bx=10*np.sin(t)
By=t**2
trajectory=np.stack((Bx,By),axis=1)
controller(magnets,trajectory,time_step=timestep,xy=np.array([0,0])[None,:])

magnets.vector_plot()
magnets.probe()
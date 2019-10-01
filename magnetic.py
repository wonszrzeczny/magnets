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
from tensorflow import signal


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
    def __init__(self,Fourier=None,time_max=1,n=1000):
        if Fourier is not None:
            self.fourier_series=Fourier*n
            self.fn=Fourier.shape[1]
            self.n=n
        self.time_max=time_max
    def randomize(self,max):
        self.fourier_series=self.n*max*(np.random.random((2,self.fn))-0.5)
        print(self.fourier_series)

    def FT(self):
        self.im_FT=np.fft.fft(self.value*1j+self.time)[0:self.fn]
        self.fourier_series=np.zeros((2,self.fn))
        self.fourier_series[0]=np.imag(self.im_FT,dtype=np.float)
        self.fourier_series[1]=-np.real(self.im_FT,dtype=np.float)
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
        complex_f=self.fourier_series.eval()
        ft=np.real(np.fft.irfft(complex_f[0]+1j*complex_f[1],n=self.n))
        ft=ft.astype(dtype=np.float)
        return ft
    def normalised_FT(self):
        return self.fourier_series/self.max













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
            self.Trajectories.append(Trajectory(fourier_coeff[i],time_max=time))
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

def loss(magnet_array):
    def value(y_true, y_pred):
        with tf.Session() as sess:
            y_pred= y_pred.eval()
            y_true=y_true.eval()
        targetu=Trajectory(y_true[0]).y()
        targetv=Trajectory(y_true[1]).y()
        actualu,actualv= magnet_array.trajectory_field(y_pred)
        plt.plot(actualu)
        plt.plot(targetu)
        plt.show()
        return np.sum((targetu-actualu)**2+ (targetv-actualv)**2)
    return value

samples=1000

def to_complex(tensor):
    return tf.complex(tensor[:,0,:],tensor[:,1,:])
def field(inputcoeff):
    const = tf.constant(inputcoeff, shape=(4,20))
    def output(tensor):
        return tensor*



model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,2,20)),
    keras.layers.Dense(160, activation='relu'),
    keras.layers.Dense(160, activation='relu'),
    keras.layers.Reshape((4,2,20)),
    keras.layers.Lambda(to_complex,output_shape=(4,20), mask=None, arguments=None),
    keras.layers.Lambda(tf.signal.irfft,output_shape=(4,20), mask=None, arguments=None),


])

magnets=Array()
magnets.orient(np.array([0,0]))
dataset=20*(np.random.random((100,2,2,20))-0.5)
optimizer = tf.keras.optimizers.RMSprop(0.001)
dupa=loss(magnets)
model.compile(loss='mean_squared_error',
              optimizer=optimizer,
              metrics=['mae', 'mse'])

model.summary()

trajectory=Trajectory(dataset[0,0]).plot_FT()
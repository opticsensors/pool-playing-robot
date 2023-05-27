import numpy as np
import matplotlib.pyplot as plt

goal_position=1462
initial_position=0
v=100
a=8
accelTime=v/a
constTime=abs(goal_position-(initial_position+a*accelTime**2))/v
endTime=2*accelTime+constTime

t_phase1=np.arange(0,accelTime,0.05)
t_phase2=np.arange(accelTime,endTime-accelTime,0.05)
t_phase3=np.arange(endTime-accelTime,endTime,0.05)

#Acceleration Phase: t = [0, accelTime]
pos_phase1=a*t_phase1**2/2
#Constant Speed Phase t = [accelTime, endTime – accelTime]
pos_phase2=v*t_phase2-v**2/(2*a)
#Deceleration Phase t = [endTime – accelTime, endTime]
pos_phase3=(2*a*v*endTime-2*v**2-a**2*(t_phase3-endTime)**2)/(2*a)

plt.plot(t_phase1,pos_phase1, 'r', linewidth=1.5)
plt.plot(t_phase2,pos_phase2, 'b', linewidth=1.5)
plt.plot(t_phase3,pos_phase3, 'g', linewidth=1.5)
plt.show()


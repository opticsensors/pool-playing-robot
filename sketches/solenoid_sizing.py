import math
import numpy as np

mass_ball=53*10**-3 #kg

# exp 1 (dist in m)
stroke_length_1=2*10**-3
incr_x_cue_ball_1=660*10**-3
time_1=2.25

# exp 2 (dist in m)
stroke_length_2=2*10**-3
dist_pool_and_cue_balls=100*10**-3
incr_x_cue_ball_2=100*10**-3
incr_x_pool_ball_2=240*10**-3

# desacceleration and mu
# using exp 1 data:
u=2*incr_x_cue_ball_1/time_1
F=0.5*mass_ball*u**2/stroke_length_1
a=-u/time_1
mu=a/9.81

# coeff of restitution
# v**2-u**2=2*a*incr_x
Uc=u
Vc=math.sqrt(Uc**2+2*a*incr_x_cue_ball_2)
Up=math.sqrt(-2*a*incr_x_pool_ball_2)
e=Up/Vc

# necessary solenoid force using desaccel and e 
# specification of incr_x_pool_ball
desired_incr_x_pool_ball=1000*10**-3
desired_stroke_length=2*10**-3
Up=math.sqrt(-2*a*desired_incr_x_pool_ball)
Vc=Up/e
Uc=math.sqrt(Vc**2-2*a*desired_incr_x_pool_ball)
F=0.5*mass_ball*Uc**2/desired_stroke_length
print(F)
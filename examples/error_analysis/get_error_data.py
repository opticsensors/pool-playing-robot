from pool.error_analysis import VisionErrorAnalysis, ActuatorErrorAnalysis

num_real_points=7000
num_estimated_points=100
maximum_error_vision_system=0.25 #units in mm

vision_error=VisionErrorAnalysis()
df=vision_error.get_error_data_vision_system(num_real_points,num_estimated_points,maximum_error_vision_system)
df.to_csv(path_or_buf=f'./results/error_data_{maximum_error_vision_system}_mm.csv', sep=',',index=False)

num_real_points=7000
number_of_angles=100
maximum_error_actuator=0.125 # units in degrees

actuator_error=ActuatorErrorAnalysis()
df=actuator_error.get_error_data_actuator(num_real_points, maximum_error_actuator, number_of_angles)
df.to_csv(path_or_buf=f'./results/error_data_{maximum_error_actuator}_degrees.csv', sep=',',index=False)


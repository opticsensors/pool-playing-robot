from pool.error_analysis import ErrorAnalysis

num_real_points=5000
num_estimated_points=15
maximum_error_vision_system=0.5 #units in mm

ea=ErrorAnalysis()
df=ea.get_error_data(num_real_points,num_estimated_points,maximum_error_vision_system)
df.to_csv(path_or_buf=f'./results/error_data_{maximum_error_vision_system}_mm.csv', sep=',',index=False)


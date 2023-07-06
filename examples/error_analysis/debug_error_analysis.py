import matplotlib.pyplot as plt
import pandas as pd
from pool.error_analysis import DebugErrorAnalysis

dea=DebugErrorAnalysis()
data = pd.read_csv('./results/error_data_0.5_mm.csv', sep=',',decimal='.')
data=data.dropna()     #drop all rows that have any NaN values


real_points_ids=[1234, 144, 1900]

for real_point_id in real_points_ids:
    fig, ax = plt.subplots()
    ax.set_xlim((0, dea.pool_table_size[0]))
    ax.set_ylim((0, dea.pool_table_size[1]))
    sub_data=data[data['real_point_id']==real_point_id]
    rows=sub_data.to_dict('records')
    ax=dea.draw_pool_table_with_pockets(ax)
    # we read from first row of sub_data, since the following rows will be the same:
    row0=rows[0]
    ax=dea.draw_ideal_configuration(ax,row0)
    ax=dea.draw_region_of_interest(ax,row0)
    for row in rows:
        ax=dea.draw_real_configuration(ax,row)
    
    ax.plot()  
    plt.show()


data = pd.read_csv('./results/error_data_0.5_degrees.csv', sep=',',decimal='.')
data=data.dropna()     #drop all rows that have any NaN values

real_points_ids=[234, 144, 1900]

for real_point_id in real_points_ids:
    fig, ax = plt.subplots()
    ax.set_xlim((0, dea.pool_table_size[0]))
    ax.set_ylim((0, dea.pool_table_size[1]))
    sub_data=data[data['real_point_id']==real_point_id]
    rows=sub_data.to_dict('records')
    ax=dea.draw_pool_table_with_pockets(ax)
    # we read from first row of sub_data, since the following rows will be the same:
    row0=rows[0]
    ax=dea.draw_ideal_configuration(ax,row0)
    ax=dea.draw_region_of_interest(ax,row0)
    for row in rows:
        ax=dea.draw_real_configuration(ax,row)
    
    ax.plot()  
    plt.show()
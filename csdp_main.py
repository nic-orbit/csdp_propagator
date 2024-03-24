import numpy as np
import matplotlib.pyplot as plt
from typing import cast
import os

import pycode.grid_eval as ge

# put the name of the orbit folder here
orbit_name = "200_550_81_100s_3months"
# choose the sampling rate (in Hz)
sample_rate = 0.01

# don't touch this :)
file_path = os.path.join("data", orbit_name)
load_path = os.path.join(file_path, orbit_name + '.npz')

satellite_orbit = ge.SatelliteOrbit()
satellite_orbit.read_from_files(load_path, sample_rate=sample_rate)

deg_to_rad = np.pi / 180

my_atm_grid = ge.AtmosphereGrid(h_low_bound=200000, h_high_bound=450000, d_h=25000, d_lat=10 * deg_to_rad, d_lon=15 * deg_to_rad)

my_atm_grid.print_info()

my_analyzer = ge.GridCoverageAnalyzer(my_atm_grid, satellite_orbit)

my_analyzer.analyze()

my_analyzer.print_statistics()

coverage_table = my_analyzer.get_statistics_dataframe()

# print only 
print(coverage_table[['Altitude Section', 'Number of Points', 'Coverage Percentage']])

my_analyzer.write_coverage_percentage_to_file(file_path)

my_analyzer.plot_grid_coverage()

my_analyzer.plot_slt_coverage()

print(set(satellite_orbit.local_times))

print(my_analyzer.d)

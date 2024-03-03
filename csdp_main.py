import numpy as np
import matplotlib.pyplot as plt
from typing import cast

import pycode.grid_eval as ge

file_path = r"data\\"

satellite_orbit = ge.SatelliteOrbit()
satellite_orbit.read_from_files(file_path)
# satellite_orbit.sample(100)

deg_to_rad = np.pi / 180

my_atm_grid = ge.AtmosphereGrid(h_low_bound=200000, h_high_bound=450000, d_h=25000, d_lat=10 * deg_to_rad, d_lon=15 * deg_to_rad)

my_atm_grid.print_info()

my_analyzer = ge.GridCoverageAnalyzer(my_atm_grid, satellite_orbit)

my_analyzer.analyze()

my_analyzer.print_statistics()

coverage_table = my_analyzer.get_statistics_dataframe()

print(coverage_table[['Number of Packages', 'Number of empty Packages', 'Coverage Percentage']])

my_analyzer.plot_grid_coverage()
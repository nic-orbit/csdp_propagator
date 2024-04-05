import numpy as np
import os

import pycode.grid_eval as ge

# put the name of the orbit folder here
orbit_name = "orbit_6month"
# choose the sampling rate (in Hz)
sample_rate = 1 / 10

# don't touch this :)
file_path = os.path.join("data", orbit_name)
load_path = os.path.join(file_path, orbit_name + ".npz")

satellite_orbit = ge.SatelliteOrbit(orbit_name)
satellite_orbit.read_from_files(load_path, sample_rate=sample_rate)

deg_to_rad = np.pi / 180

my_atm_grid = ge.AtmosphereGrid(
    h_low_bound=200000,
    h_high_bound=450000,
    d_h=25000,
    d_lat=10 * deg_to_rad,
    d_lon=15 * deg_to_rad,
)

my_atm_grid.print_info()

my_analyzer = ge.GridCoverageAnalyzer(my_atm_grid, satellite_orbit)

my_analyzer.analyze()

my_analyzer.print_statistics()

coverage_table = my_analyzer.get_statistics_dataframe()

# print only
print(coverage_table[["Altitude Section", "Number of Points", "Coverage Percentage"]])

my_analyzer.write_coverage_percentage_to_file(file_path)

my_analyzer.plot_grid_coverage()

my_analyzer.plot_slt_coverage()

print(
    "fraction of time below 450 km: ",
    satellite_orbit.calculate_fraction_below_altitude(450000),
)

print(
    "time below 450 km: ",
    satellite_orbit.calculate_fraction_below_altitude(450000) * (5 + 23 / 30) * 3,
    (5 + 23 / 30) * 3,
)

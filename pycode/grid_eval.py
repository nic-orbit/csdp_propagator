import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import cast
import pandas as pd
import os

class AtmospherePackage:
    """
    A class used to represent an Atmosphere Package.

    ...

    Attributes
    ----------
    h_min : float
        minimum height of the atmosphere package
    h_max : float
        maximum height of the atmosphere package
    lat_min : float
        minimum latitude of the atmosphere package
    lat_max : float
        maximum latitude of the atmosphere package
    lon_min : float
        minimum longitude of the atmosphere package
    lon_max : float
        maximum longitude of the atmosphere package
    points : np.array
        array of points in the atmosphere package
    number_of_points : int
        total number of points in the atmosphere package
    index : tuple
        current index in the atmosphere package

    Methods
    -------
    __init__(self, h_min:float, h_max:float, lat_min:float, lat_max:float, lon_min:float, lon_max:float, points:np.array) -> None:
        Initializes the AtmospherePackage with the given parameters.
    """
    def __init__(self, h_min:float, h_max:float, lat_min:float, lat_max:float, lon_min:float, lon_max:float, points:np.array) -> None:
        self.h_max = h_max
        self.h_min = h_min
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.points = points
        self.number_of_points = points.shape[0]
        self.index = (0,0)
        self.solar_time_coverage()

    def info(self):
        print(f'AtmospherePackage: \n {self.h_min} to {self.h_max} km; \n {self.lat_min} to {self.lat_max} rad; \n {self.lon_min} to {self.lon_max} rad; \n {self.number_of_points} points; \n first point timestamp: {self.points[0][0]} \n'
              + 'Solar time coverage: ' + str(self.slts_covered) + '\n' + 'Coverage percentage: ' + str(self.slts_coverage_percentage) + '%\n'
              + 'times: ' + str(self.points[:, 7]))

    def solar_time_coverage(self):
        times = set(self.points[:, 7])
        self.slts_covered = times
        self.slts_coverage_percentage = len(times) / 24 * 100
        # print('times covered: ', times)
        # print('number of times covered: ', len(times))


class AltitudeSection:
    
    def __init__(self, h_min:float, h_max:float, points:np.array) -> None:
        self.h_min = h_min
        self.h_max = h_max
        self.points = points
        self.number_of_points = points.shape[0]
        self.packages = []
        self.coverage_matrix = np.array([None])
        self.coverage_percentage = 0
        self.number_of_empty_packages = 0
        self.d_lat = 0
        self.d_lon = 0
        self.slt_count_std = 0
        self.slt_count_avg = 0
        self.slts = set()
        self.description = f'Altitude section from {h_min} to {h_max} km'
    
    def sort_points_into_packages(self, d_lat:float, d_lon:float):
        print('sorting points into packages ...')
        self.d_lat = d_lat
        self.d_lon = d_lon
        number_of_latitude_sections = int(np.pi / d_lat) # per altitude section
        self.number_of_latitude_sections = number_of_latitude_sections
        number_of_longitude_sections = int(2*np.pi / d_lon) # per altitude section and latitude section
        self.number_of_longitude_sections = number_of_longitude_sections

        for i in range(number_of_latitude_sections):
            # latitude from pi/2 to -pi/2 or 90 to -90 deg
            lat_max = np.pi/2 - i * d_lat
            lat_min = np.pi/2 - (i + 1) * d_lat
            # print(lat_min, lat_max)
            for j in range(number_of_longitude_sections):
                # longitude from -pi to pi or -180 to 180 deg
                lon_min = -np.pi + j * d_lon
                lon_max = -np.pi + (j + 1) * d_lon
                
                points = self.points[(self.points[:, 4] >= lat_min) & (self.points[:, 4] < lat_max) & (self.points[:, 5] >= lon_min) & (self.points[:, 5] < lon_max)]

                # print('  ', lon_min, lon_max, '---', points[:,7])

                atm_p = AtmospherePackage(self.h_min, self.h_max, lat_min, lat_max, lon_min, lon_max, points)
                atm_p.index = (i, j) 
                self.packages.append(atm_p)

    def calculate_coverage_matrix(self):

        print('calculating point density matrix')

        if self.d_lat == 0 or self.d_lon == 0:
            raise ValueError('Latitude and longitude increments must be set before calculating point density matrix.  \nRun sort_points_into_packages() method first.')

        # create matrix with number of points in each package
        data = np.zeros((self.number_of_latitude_sections, self.number_of_longitude_sections))
        for i in range(self.number_of_latitude_sections):
            for j in range(self.number_of_longitude_sections):
                data[i, j] = self.packages[i * self.number_of_longitude_sections + j].number_of_points
        
        self.coverage_matrix = data.astype(int)
        # count packages that have at least one point
        self.number_of_empty_packages = len([p for p in self.packages if p.number_of_points == 0])
        self.coverage_percentage = np.count_nonzero(data) / (len(self.packages)) * 100
        print( f'Coverage percentage: {self.coverage_percentage}%')

    def calculate_slt_statistics(self):
        slt_count_avg = []
        for pkg in self.packages:
            slt_count_avg.append(len(pkg.slts_covered))
            # print('  ', pkg.index, '---', len(pkg.slts_covered), '---', pkg.slts_covered)
            self.slts = self.slts.union(pkg.slts_covered)
            # print('  ', self.slts)
        
        self.slt_count_std = np.round(np.std(slt_count_avg), 3)
        self.slt_count_avg = np.round(np.mean(slt_count_avg), 1)
    
    def plot_coverage(self, show_plot:bool=True):

        deg_to_rad = np.pi / 180

        if self.coverage_matrix.all() == None:
            self.calculate_coverage_matrix()
        
        data = self.coverage_matrix

        fig = plt.figure()

        # Create the plot
        plt.imshow(data, cmap='viridis', interpolation='nearest')

        # Add color bar
        plt.colorbar()

        # Label x-axis with longitude
        plt.xticks(np.arange(self.number_of_longitude_sections), [f'{int(-180 + (i * self.d_lon / deg_to_rad))} to {int(-180 + ((i+1) * self.d_lon / deg_to_rad))}' for i in range(self.number_of_longitude_sections)], rotation=-90)

        # Label y-axis with latitude
        plt.yticks(np.arange(self.number_of_latitude_sections), [f'{int(90 - (i * self.d_lat / deg_to_rad))} to {int(90 - ((i+1) * self.d_lat / deg_to_rad))} ' for i in range(self.number_of_latitude_sections)])

        # Add labels and title
        plt.xlabel('Longitude in degrees')
        plt.ylabel('Latitude in degrees')
        plt.title(f'Point density at altitude {self.h_min / 1000} to {self.h_max / 1000} km; \n {self.points.shape[0]} points.')

        # Add vertical line at 0 degree longitude (greenwich meridian)
        plt.axvline(x=(self.number_of_longitude_sections-1) / 2 , color='red', linestyle='--')

        # Add horizontal line at equator
        plt.axhline(y=(self.number_of_latitude_sections-1) / 2, color='red', linestyle='--')

        plt.tight_layout()

        # Show the plot
        if show_plot:
            plt.show()


class AtmosphereGrid:

    def __init__(self, h_low_bound, h_high_bound, d_h, d_lat, d_lon) -> None:
        self.h_low_bound = h_low_bound
        self.h_high_bound = h_high_bound
        self.d_h = d_h
        self.d_lat = d_lat
        self.d_lon = d_lon
        
        self.number_of_altitude_sections = int((h_high_bound - h_low_bound) / d_h)
        self.number_of_latitude_sections = int(np.pi / d_lat)
        self.number_of_longitude_sections = int(2*np.pi / d_lon)
        self.number_of_packages = self.number_of_altitude_sections * self.number_of_latitude_sections * self.number_of_longitude_sections
        print('atmosphere grid created')
    
    def print_info(self):
        print(f'Atmosphere grid from {self.h_low_bound} to {self.h_high_bound} km altitude.')
        print(f'Number of altitude sections: {self.number_of_altitude_sections}')
        print(f'Number of latitude sections: {self.number_of_latitude_sections}')
        print(f'Number of longitude sections: {self.number_of_longitude_sections}')
        print(f'Number of packages: {self.number_of_packages}')
      

class SatelliteOrbit:
    def __init__(self) -> None:
        pass

    def generate_random_data(self, k:int):
        # Generating random data for demonstration
        self.timestamps = np.linspace(0, 3600, k)  # k timestamps evenly spaced over 3600 hours
        self.x_coords = np.random.uniform(-1000, 1000, k)
        self.y_coords = np.random.uniform(-1000, 1000, k)
        self.z_coords = np.random.uniform(-100, 100, k)
        self.latitudes = np.random.uniform(-70, 85, k)
        self.longitudes = np.random.uniform(-180, 180, k)
        self.altitudes = np.random.uniform(100, 600, k)

    def read_from_files(self, load_path:str, sample_rate:int = 0):
        loaded = np.load(load_path)
        self.timestamps = loaded['time'] # in hours
        self.x_coords = np.zeros(len(self.timestamps))
        self.y_coords = np.zeros(len(self.timestamps))
        self.z_coords = np.zeros(len(self.timestamps))
        self.latitudes = loaded['lat']
        self.longitudes = loaded['lon']
        self.altitudes = loaded['alt']
        self.local_times = np.zeros(len(self.timestamps))

        original_sample_rate =  len(self.timestamps) / ((self.timestamps[-1] - self.timestamps[0]) * 3600 ) # in Hz
        print(f'Original sample rate: {original_sample_rate} Hz')
        if sample_rate > original_sample_rate:
            raise ValueError('Sample rate cannot be higher than original sample rate.')
        
        if sample_rate == 0:
            print('No sampling')
        else:
            print(f'Sampling at {sample_rate} Hz')
            self.timestamps = self.timestamps[::int(original_sample_rate / sample_rate)]
            self.x_coords = self.x_coords[::int(original_sample_rate / sample_rate)]
            self.y_coords = self.y_coords[::int(original_sample_rate / sample_rate)]
            self.z_coords = self.z_coords[::int(original_sample_rate / sample_rate)]
            self.latitudes = self.latitudes[::int(original_sample_rate / sample_rate)]
            self.longitudes = self.longitudes[::int(original_sample_rate / sample_rate)]
            self.altitudes = self.altitudes[::int(original_sample_rate / sample_rate)]
            self.local_times = np.zeros(len(self.timestamps))

        self.calculate_local_time()
        self.create_points_array()
        
        

    def create_points_array(self):
        self.points = np.column_stack((self.timestamps, self.x_coords, self.y_coords, self.z_coords, self.latitudes, self.longitudes, self.altitudes, self.local_times))
        print('satellite orbit created')

    def calculate_local_time(self):
        # Calculate (relative) local time from longitude; 
        # start at 0 hours and 0 degree longitude (greenwich meridian) is used here (irrelevant for this project, since relative local time is used)
        deg_to_rad = np.pi / 180

        # Calculate solar local time for each longitude
        for i in range(len(self.timestamps)):
            slt = np.round((self.timestamps[i] + (self.longitudes[i] / (15 * deg_to_rad))) % 24, 0) # solar local time
            # print(slt)
            if slt == 24:
                slt = 0

            self.local_times[i] = int(slt)

    def sample(self, k:int):
        self.points = self.points[::k]
        print('satellite orbit sampled')

    def plot_altitude_slt_scatter(self):
        plt.plot(self.local_times, self.altitudes / 1000, marker='o', linestyle='none')
        plt.xlabel('Local Solar Time [hours]')
        plt.ylabel('Altitude [km]')
        plt.title('Altitude over Local Solar Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        

class GridCoverageAnalyzer:

    def __init__(self, atm_grid:AtmosphereGrid, orbit:SatelliteOrbit) -> None:
        self.atm_grid = atm_grid
        self.orbit = orbit
        self.points = orbit.points
        self.list_of_altitude_sections = []
        print('grid coverage analyzer created')

    def analyze(self):
        print('starting analysis')
        for i in range(self.atm_grid.number_of_altitude_sections):
        # altitude from lowest to highest
            print(f'Analyzing altitude section {i+1} of {self.atm_grid.number_of_altitude_sections}')
            h_min =  self.atm_grid.h_low_bound + i * self.atm_grid.d_h
            h_max = self.atm_grid.h_low_bound + (i + 1) * self.atm_grid.d_h
            points = self.points[(self.points[:, 6] >= h_min) & (self.points[:, 6] < h_max)]
            altitude_sec = AltitudeSection(h_min, h_max, points)
            altitude_sec.sort_points_into_packages(self.atm_grid.d_lat, self.atm_grid.d_lon)
            altitude_sec.calculate_coverage_matrix()
            altitude_sec.calculate_slt_statistics()
            self.list_of_altitude_sections.append(altitude_sec)

    def print_statistics(self):
        print('Printing statistics ...')
        for i, altitude_section in enumerate(self.list_of_altitude_sections):
            print(f'Altitude section {i+1} of {len(self.list_of_altitude_sections)}')
            print(f'  Number of points: {altitude_section.points.shape[0]}')
            print(f'  Number of packages: {len(altitude_section.packages)}')
            print(f'  Number of latitude sections: {altitude_section.number_of_latitude_sections}')
            print(f'  Number of longitude sections: {altitude_section.number_of_longitude_sections}')
            print(f'  Coverage percentage: {altitude_section.coverage_percentage}%')
            print(f'  Average number of Solar local times covered per Package: {altitude_section.slt_count_avg} +/- {altitude_section.slt_count_std} hours')
            print(f'  Solar times covered: {altitude_section.slts}')
            print('')

    def get_statistics_dataframe(self):
        data = {
            'Altitude Section': [],
            'Number of Points': [],
            'Number of Packages': [],
            'Number of empty Packages': [],
            'Number of Latitude Sections': [],
            'Number of Longitude Sections': [],
            'Coverage Percentage': []
            }
    
        for i, altitude_section in enumerate(self.list_of_altitude_sections):
            data['Altitude Section'].append(f" {altitude_section.h_min / 1000} to {altitude_section.h_max / 1000} km")
            data['Number of Points'].append(altitude_section.points.shape[0])
            data['Number of Packages'].append(len(altitude_section.packages))
            data['Number of empty Packages'].append(altitude_section.number_of_empty_packages)
            data['Number of Latitude Sections'].append(altitude_section.number_of_latitude_sections)
            data['Number of Longitude Sections'].append(altitude_section.number_of_longitude_sections)
            data['Coverage Percentage'].append(altitude_section.coverage_percentage)
        
        df = pd.DataFrame(data)
        return df
    
    def write_coverage_percentage_to_file(self, file_path:str):
        df = self.get_statistics_dataframe()
        df_filtered = df[['Altitude Section', 'Coverage Percentage']].transpose(copy=True)
        df_filtered.to_csv(os.path.join(file_path,'coverage_percentage.csv'), index=False)
        print('Coverage percentage written to file')

    def plot_grid_coverage(self, onebyone:bool=False):
        if len(self.list_of_altitude_sections) == 0:
            raise ValueError('No altitude sections to plot.  \nRun analyze() method first.')
        for i, altitude_section in enumerate(self.list_of_altitude_sections):

            print(f'Plotting altitude section {i+1} of {len(self.list_of_altitude_sections)}')
            altitude_section.plot_coverage(show_plot=onebyone)
        plt.show() 

    def plot_slt_coverage(self):
        # Plot altitude sections over solar local time
        
        self.d = np.zeros((len(self.list_of_altitude_sections), 24))
        
        for i, altitude_section in enumerate(self.list_of_altitude_sections):
            # print(altitude_section.slts)
            for j in altitude_section.slts:
                # print(j)
                self.d[i, int(j)] = 1
                
        self.d = np.flipud(self.d) # flip matrix upside down to have lowest altitude at the bottom

        fig, ax = plt.subplots()
        im = ax.imshow(self.d, cmap='viridis', interpolation='nearest')

        # Add legend annotations
        # legend_elements = [plt.Line2D([0], [0], marker='s', color='w', label='Visited', markerfacecolor='yellow', markersize=10),
        #            plt.Line2D([0], [0], marker='s', color='w', label='Not Visited', markerfacecolor='purple', markersize=10)]
        # get the colors of the values, according to the 
        # colormap used by imshow
        values = np.unique(self.d.flatten())       

        if self.d.sum() == len(self.d.flatten()):
            colors = [ "yellow"]
            labels = [ "Visited"]
        elif self.d.sum() == 0:
            colors = [ "purple"]
            labels = [ "Not Visited"]
        else:
            colors = [ "purple", "yellow"]
            labels = [ "Not Visited", "Visited"]

        colors = [ im.cmap(im.norm(value)) for value in values]

        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], label=labels[i] ) for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        ax.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)


        plt.yticks(np.arange(len(self.list_of_altitude_sections)), [f'{int(altitude_section.h_min / 1000)} to {int(altitude_section.h_max / 1000)} km' for altitude_section in reversed(self.list_of_altitude_sections)])
        ax.set_xlabel('Solar Local Time')
        ax.set_ylabel('Altitude Section')
        plt.title('Solar Local Time Coverage')
        # plt.colorbar(im)  # Add colorbar
        plt.tight_layout()
        plt.show()


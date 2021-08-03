from logging import warning

import pandas as pd

from numpy import fft
import numpy as np
import matplotlib.pyplot as plt


from curlypiv.datasets.standard_dataset import StandardDataset
from curlypiv.datasets.bpe_iceo_dataset import BpeIceoDataset
from curlypiv.datasets.bpe_iceo_fluoroquench_dataset import BpeIceoFluoroQuenchDataset
from curlypiv.metrics import DatasetMetric, BpeIceoDatasetMetric, utils
from curlypiv.statistical_analysis.signal_processing import generate_waveform


class BpeIceoFluoroQuenchDatasetMetric(BpeIceoDatasetMetric):
    """
    Class for computing BPE-ICEO fluorescein quenching related metrics on a single :obj: '~curlypiv.datasets.BpeIceoFluoroQuenchDataset'
    """

    def __init__(self, dataset, fluorescein_quenching_data_filepath, electric_field_strength=10e3, frequency=100, waveform='square'):
        """
        Args:
            dataset (StandardDataset): A StandardDataset.
            electric_field_strength (int): an example electric field strength to compute basic stats.
            frequency (int): an example frequency to compute basic stats

        Raise:
            TypeError: 'dataset' must be a :obj: '~curlypiv.datasets.BpeIceoActuatorDataset' type.
        """
        if not isinstance(dataset, BpeIceoFluoroQuenchDataset):
            raise TypeError("'dataset' should be a BpeIceoActuatorDataset")

        # sets the self.dataset and frequency
        super(BpeIceoFluoroQuenchDatasetMetric, self).__init__(dataset, electric_field_strength=electric_field_strength,
                                                           frequency=frequency, waveform=waveform)
        self.filepath = particle_tracking_data_filepath

        # some basic derived terms
        #   membrane stiffness
        #   pressure load required to cause the membrane to displace 5 microns

        # 1. pre-process results
        self.df = self.preprocess_particle_tracking()
        #       read actuator_data_filepath (can be either filepath or pandas dataframe) into a dataframe.
        #       get rid of any rows with Nans.
        #       make sure the units are correct.
        #       get the mean value of all particles on the first frame and subtract from all particles (initialize z=0)

        # 2. analyze results
        self.dff = self.analyze_particle_tracking(num_inspection_particles=9)
        #       define a "number of particles to inspect" variable (i.e. to plot initially)
        #       for each particle, analyze the signal variance.
        #       if the variance is too high, filter out the particle (by creating a new "reduced" dataframe)
        #       if # particles > # inspect particles, sort particles by # of images and take most reoccurring particles.

        # 3. plot results
        #self.visualize_particle_tracking()
        #       plot the original dataset for all particles in all images.
        #       plot the reduced dataset for filtered particles in all images.

    def __repr__(self):
        class_ = 'BpeIceoActuatorDatasetMetric'
        repr_dict = {
            'Geometry': self.geometry,
            'Characteristic': self.characteristic,
            'Capacitance': self.capacitance,
            'Current': self.current,
            'Resistance': self.resistance,
            'Timescale': self.timescale,
            'Induced Zeta': self.zeta_induced,
            'Slip Velocity': self.u_slip
        }
        out_str = "{}: \n".format(class_)
        for key, val in repr_dict.items():
            if type(val) is list:
                print('list')
            elif type(val) is dict:
                for keyy, vall in val.items():
                    if type(vall) is np.ndarray:
                        pass
                    else:
                        out_str += '{}: {} \n'.format(keyy, str(vall))
            else:
                out_str += '{}: {} \n'.format(key, str(val))
        return out_str

    def preprocess_particle_tracking(self):
        """
        pre process steps:
            read actuator_data_filepath (can be either filepath or pandas dataframe) into a dataframe.
            get rid of any rows with Nans.
            make sure the units are correct and add time column.
            get the mean value of all particles on the first frame and subtract from all particles (initialize z=0)
        """
        # if filepath is type(dataframe):
        #   not implemented yet

        # read into dataframe
        df = pd.read_csv(self.filepath, header=0, index_col=0,
                         dtype={'Frame': int, 'id': int, 'x': float, 'y': float, 'z': float, 't': float})

        # get rid of Nans
        df.dropna(inplace=True)

        # adjust units to meaningful scale and add time column
        df.insert(loc=1, column='t', value=df.Frame/self.dataset.ccd_img_acq_rate)

        # normalize particles height by mean height
        z_mean_initial = df.loc[df['Frame'] == 0].z.mean()
        df.z = df.z - z_mean_initial

        return df

    def analyze_particle_tracking(self, num_inspection_particles=20):
        """
        analysis steps:
                define a "number of particles to inspect" variable (i.e. to plot initially)
                for each particle, analyze the signal variance.
                if the variance is too high, filter out the particle (by creating a new "reduced" dataframe)
                if # particles > # inspect particles, sort particles by # of images and take most reoccurring particles.
        """
        # define "number of inspection particles"
        #   --> defined now as an input

        # for each particle, analyze the variance

        # if the particle count is still too high, get the particle id's of the most occuring particles
        groupby_id = self.df.groupby('id').z.count()
        groupby_id.sort_values(ascending=False, inplace=True)
        sort_list = groupby_id.index.tolist()
        boolean_series = self.df.id.isin(sort_list)
        dff = self.df[boolean_series]
        if len(dff.id.unique()) > num_inspection_particles:
            num_filter_list = sort_list[:num_inspection_particles]
            boolean_num_filt_series = dff.id.isin(num_filter_list)
            dff = dff[boolean_num_filt_series]

        return dff


    def visualize_particle_tracking(self):
        """
        visualization steps:
                plot the original dataset for all particles in all images.
                plot the reduced dataset for filtered particles in all images.
                box plot of filtered particles

        Some useful references:
            https://scipy-lectures.org/packages/statistics/index.html
        """
        # simple line plot
        self.plot_rolling_window_line_plot()
        self.plot_applied_signal_convolution_line_plot()
        self.plot_box_plot()


    # ----- SIGNAL PROCESSING -----
    def convolve_with_applied_signal(self):
        """
        Generate the input waveform and (eventually, convolve with signal -- not implemented yet).
        Inputs:
            function (): 'sine' or 'square'
            frequency (Hz): frequency of the returned waveform.
            length (#): number of samples in the returned waveform.
            sampling_rate (Hz): sampling frequency of the returned waveform.
            amplitude (): always return an amplitude of 1 that can be scaled for each application.
            theta (rad): specify initial value of signal using theta.
        Returns:
            generated waveform (ndarray):
        """
        applied_waveform = generate_waveform(function=self.waveform, frequency=self.frequency, length=len(self.dff.t.unique()),
                                             sampling_rate=self.dataset.ccd_img_acq_rate, amplitude=1, theta=0)

        return applied_waveform

    def plot_rolling_window_line_plot(self, rolling_window=24):
        plt.figure()
        for i in self.dff.id.unique():
            x = self.dff.loc[self.dff['id'] == i].t.to_numpy(dtype=float)
            y = self.dff.loc[self.dff['id'] == i].z
            y = y.rolling(window=rolling_window, min_periods=rolling_window, center=True).mean().to_numpy(dtype=float)
            plt.plot(x, y)

        plt.xlabel('time (s)')
        plt.ylabel(r'displacement ($\mu$ m)')
        plt.title('Particle tracjetories (avg of ~1 s window)'.format(rolling_window))
        plt.show()

    def plot_applied_signal_convolution_line_plot(self, rolling_window=10):
        plt.figure()

        # plot applied signal
        applied_waveform = self.convolve_with_applied_signal()
        plt.plot(applied_waveform[0, :], applied_waveform[1, :], color='lightgray', alpha=0.35, label='applied waveform')

        # plot particle displacements
        for i in self.dff.id.unique():
            x = self.dff.loc[self.dff['id'] == i].t.to_numpy(dtype=float)
            y = self.dff.loc[self.dff['id'] == i].z
            y = y.rolling(window=rolling_window, min_periods=rolling_window, center=True).mean().to_numpy(dtype=float)
            plt.plot(x, y)

        plt.xlabel('time (s)')
        plt.ylabel(r'displacement ($\mu$ m)')
        plt.suptitle('Particle trajectories w/ applied {} wave'.format(self.waveform))
        plt.title('E: {} V/mm, f: {} Hz, Avg: {} samples'.format(self.electric_field_strength, self.frequency, rolling_window), fontsize=11)
        plt.legend(loc='lower left')
        plt.show()

    def plot_box_plot(self):
        groupby_id = self.dff.groupby('id')
        boxplot = groupby_id.boxplot(column=['z'])
        #boxplot = self.dff.boxplot(column=['z'], ax=ax)
        plt.show()
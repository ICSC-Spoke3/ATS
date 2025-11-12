import datetime as dt
import pandas as pd
import random as rnd
import numpy as np
import pytz
import math
from copy import deepcopy


def _quantities_in(timeseries):
    quantities = list(timeseries.columns)
    if 'timestamp' in quantities:
        quantities.remove('timestamp')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')
    return quantities


def generate_time_boundaries(time_interval='90D',starting_year=None,
                             starting_month= None,starting_day=None,
                             starting_hour=None,starting_minute=None):

    time_span = pd.Timedelta(time_interval).to_pytimedelta()
    datetime_boundaries = []

    years = list(range(1970,2025))
    months = list(range(1,13))
    days = list(range(1,9))
    hours = list(range(0,24))
    minutes = list(range(0,60))
    starting_year = starting_year if starting_year is not None else rnd.choice(years)
    starting_month = starting_month if starting_month is not None else rnd.choice(months)
    starting_day = starting_day if starting_day is not None else rnd.choice(days)
    starting_hour = starting_hour if starting_hour is not None else rnd.choice(hours)
    starting_minute = starting_minute if starting_minute is not None else rnd.choice(minutes)
    starting_datetime = dt.datetime(starting_year,starting_month,starting_day,starting_hour,
                                        starting_minute,tzinfo=pytz.UTC)
    final_datetime = starting_datetime + time_span

    datetime_boundaries.append(starting_datetime)
    datetime_boundaries.append(final_datetime)

    return datetime_boundaries


def generate_synthetic_humitemp_timeseries(sampling_interval,time_boundaries=[],
                                           temperature=True,humidity=True):
    if not isinstance(temperature,bool):
        raise TypeError(f'"temperature" must be of type bool, got {type(temperature).__name__} instead.')

    if not isinstance(humidity,bool):
        raise TypeError(f'"humidity" must be of type bool, got {type(humidity).__name__} instead.')

    if not isinstance(sampling_interval,dt.timedelta):
        raise TypeError(f'"sampling_interval" must be of type datetime.timedelta, got {type(sampling_interval).__name__} instead.')

    if not isinstance(time_boundaries,list):
        raise TypeError(f'"time_boundaries" must be of type list, got {type(time_boundaries).__name__} instead.')

    if not time_boundaries:
        raise ValueError('Invalid argument: "time_boundaries" must be a non empty list')

    if time_boundaries[0] > time_boundaries[1]:
        raise ValueError('Invalid argument: in "time_boundaries", "{}" occurs after "{}". Ensure the timestamp values are in chronological order'.format(time_boundaries[0],time_boundaries[1]))

    temp = []
    humi = []
    timestamp = []
    anomaly_label = []
    effect_label = []
    data_points = {}

    min_temp = 10
    max_temp = 25
    temperature_amplitude = (max_temp - min_temp)/2
    min_humi = 50
    max_humi = 90
    humidity_amplitude = (max_humi - min_humi)/2
    periodicity = 24 #24 h 
    delta_t = pd.Timedelta(sampling_interval).to_pytimedelta()
    time_value = time_boundaries[0]

    while time_value < time_boundaries[1]:
        anomaly_label.append(None)
        effect_label.append(None)
        timestamp.append(time_value)

        time_variable = time_value.hour + time_value.minute/60
        sin_value = math.sin((2*math.pi/periodicity)*(time_variable-6))

        temp.append(min_temp + temperature_amplitude * (1 + sin_value))
        humi.append(min_humi + humidity_amplitude * (1 - sin_value))

        time_value = time_value + delta_t

        if time_value == time_boundaries[1]:
            break

    data_points.update({'timestamp': timestamp})
    data_points.update({'anomaly_label': anomaly_label})
    data_points.update({'effect_label': effect_label})

    if temperature:
        data_points.update({'temperature': temp})

    if humidity:
        data_points.update({'humidity': humi})

    if not temperature and not humidity:
        raise ValueError('Error: no data selected for creating the DataFrame. Set at True at least one of the two arguments: temperature or humidity')

    return pd.DataFrame(data_points)


# Spike anomaly
def add_spike_anomaly(timeseries,inplace=False,mode='uv'):
    if not inplace:
        timeseries = deepcopy(timeseries)
    quantities = _quantities_in(timeseries)

    spike_intensities = {'low': 5,
    'medium':7,
    'high':9
    }
    anomalous_spike_position = 10
    for i in range(anomalous_spike_position,len(timeseries)-anomalous_spike_position):
        data_point_effect = timeseries.loc[i,'effect_label']
        if data_point_effect is not None and 'spike' in data_point_effect:
            anomalous_spike_position +=1
        else:
            break
    intensity = rnd.choice(list(spike_intensities.keys()))

    if mode == 'uv':
        timeseries.loc[anomalous_spike_position,'anomaly_label'] = 'spike_uv'
        if 'temperature' in quantities:
            timeseries.loc[anomalous_spike_position,'temperature'] -= spike_intensities[intensity]
        if 'humidity' in quantities:
            timeseries.loc[anomalous_spike_position,'humidity'] += spike_intensities[intensity]

    if mode == 'mv':
        timeseries.loc[anomalous_spike_position,'anomaly_label'] = 'spike_mv'
        if 'temperature' in quantities and 'humidity' in quantities:
            timeseries.loc[anomalous_spike_position,'temperature'] -= spike_intensities[intensity]
            timeseries.loc[anomalous_spike_position,'humidity'] += spike_intensities[intensity]
        else:
            raise ValueError('Cannot insert multivariate anomaly on a one dimensional timeseries')

    return timeseries


# Step anomaly
def add_step_anomaly(timeseries,mode='uv',inplace=False):
    if not inplace:
        timeseries = deepcopy(timeseries)
    quantities = _quantities_in(timeseries)

    ramp_height = 10
    ramp_length = 50
    step_length = int(len(timeseries)/5)
    start = 3*int(len(timeseries)/4)
    stop = min(start + step_length + (2 * ramp_length),len(timeseries))

    def compute_ramp_factor(i,current_position_in_ramp):
        if i < start + ramp_length:
            current_position_in_ramp +=1
            ramp_factor =  current_position_in_ramp  / ramp_length

        elif i < start + ramp_length + step_length:
            ramp_factor = 1

        elif i < stop:
            current_position_in_ramp -=1
            ramp_factor = current_position_in_ramp  / ramp_length

        ramp_parameters = [ramp_factor,current_position_in_ramp]

        return ramp_parameters

    if 'temperature' in quantities and 'humidity' in quantities:

        if mode == 'uv':
            current_position_in_ramp = 0

            for i in range(start,stop):
                ramp_factor,current_position_in_ramp = compute_ramp_factor(i,current_position_in_ramp)
                timeseries.loc[i,'temperature'] += (ramp_factor * ramp_height)
                timeseries.loc[i,'humidity'] -= (ramp_factor * ramp_height)
                timeseries.loc[i,'anomaly_label'] = 'step_uv'

        if mode == 'mv':
            current_position_in_ramp = 0

            for i in range(start,stop):
                ramp_factor,current_position_in_ramp = compute_ramp_factor(i,current_position_in_ramp)
                timeseries.loc[i,'temperature'] += (ramp_factor * ramp_height)
                timeseries.loc[i,'humidity'] += (ramp_factor * ramp_height)
                timeseries.loc[i,'anomaly_label'] = 'step_mv'

    else:

        if mode == 'mv':
            raise ValueError(f'Cannot generate a multivariate anomaly if there is only one quantity in the pandas.DataFrame')

        if mode == 'uv':
            current_position_in_ramp = 0

            for i in range(start,stop):
                ramp_factor,current_position_in_ramp = compute_ramp_factor(i,current_position_in_ramp)

                if quantities[0] == 'temperature':
                    timeseries.loc[i,quantities[0]] += (ramp_factor * ramp_height)

                else:
                    timeseries.loc[i,quantities[0]] -= (ramp_factor * ramp_height)

                timeseries.loc[i,'anomaly_label'] = 'step_uv'

    return timeseries


# Noise anomaly
def add_anomalous_noise(timeseries,inplace=False,mode='uv'):
    if not inplace:
        timeseries = deepcopy(timeseries)
    quantities = _quantities_in(timeseries)

    def insert_anomalous_noise(quantity,mode):
        pattern = [3, -3]
        start = int(len(timeseries)/5)
        anomalous_noise_length = int(len(timeseries)/15)
        stop = min(start + anomalous_noise_length,len(timeseries))

        for i in range(start,stop):
            timeseries.loc[i,quantity] += pattern[i % len(pattern)]
            timeseries.loc[i,'anomaly_label'] = 'noise' + '_' + mode

    if 'temperature' in quantities and 'humidity' in quantities:
        
        if mode == 'uv':
            insert_anomalous_noise('temperature','uv')
            insert_anomalous_noise('humidity','uv')

        elif mode == 'mv':
            insert_anomalous_noise('temperature','mv')

    else:

        if mode == 'uv':
            insert_anomalous_noise(quantities[0],'uv')

        elif mode == 'mv':
            raise ValueError(f'Anomalies in multivariate mode cannot be added if there is only one quantity')

    return timeseries


# Pattern anomaly
def add_pattern_anomaly(timeseries,sampling_interval,inplace=False,mode='uv'):
    if not inplace:
        timeseries = deepcopy(timeseries)
    quantities = _quantities_in(timeseries)

    start = int(len(timeseries)/3)
    anomalous_pattern_length = int(len(timeseries)/5)#piece of series with anomalous periodicity
    transition_length = 50
    stop = min(start + anomalous_pattern_length + (2 * transition_length),len(timeseries))
    anomaly_start_position = start + transition_length
    anomaly_stop_position = anomaly_start_position + anomalous_pattern_length

    def change_the_timeseries_period(quantity,start,stop,mode):
        normal_periodicity_in_hour = 24
        anomalous_periodicity_in_hour = 48
        min_quantity = min(timeseries[quantity])
        max_quantity = max(timeseries[quantity])
        quantity_amplitude = (max_quantity - min_quantity)/2

        delta_t = sampling_interval.total_seconds()/3600
        t0 = timeseries.loc[0,'timestamp']
        time_variable = t0.hour + t0.minute/60
        cumulative_phase = (2*math.pi/normal_periodicity_in_hour)*(time_variable-6)

        for i in range(len(timeseries)):

            if i < start or i > stop:
                periodicity = normal_periodicity_in_hour

            elif i > anomaly_start_position and i < anomaly_stop_position:
                periodicity = anomalous_periodicity_in_hour
                timeseries.loc[i,'anomaly_label'] = 'pattern' + '_' + mode

            else:
                timeseries.loc[i,'anomaly_label'] = 'pattern' + '_' + mode

                if i < anomaly_start_position:
                    transition_parameter = (i - (anomaly_start_position - transition_length)) / transition_length

                else:
                    transition_parameter = 1 - (i - anomaly_stop_position) / transition_length

                transition_parameter = max(0, min(1, transition_parameter))
                periodicity = (1 - transition_parameter) * normal_periodicity_in_hour + transition_parameter * anomalous_periodicity_in_hour

            phase = ((2 * math.pi)/periodicity) * delta_t
            cumulative_phase += phase
            sin_value = math.sin(cumulative_phase)

            if quantity == 'temperature':
                timeseries.loc[i,'temperature'] = min_quantity + quantity_amplitude * (1 + sin_value)

            if quantity == 'humidity':
                timeseries.loc[i,'humidity'] = min_quantity + quantity_amplitude * (1 - sin_value)

    if 'temperature' in quantities and 'humidity' in quantities:

        if mode == 'uv':
            change_the_timeseries_period('temperature',start,stop,'uv')
            change_the_timeseries_period('humidity',start,stop,'uv')

        if mode == 'mv':
            change_the_timeseries_period('temperature',start,stop,'mv')

    else:

        if mode == 'uv':
            change_the_timeseries_period(quantities[0],start,stop,'uv')

        if mode == 'mv':
            raise ValueError(f'Cannot generate a multivariate anomaly if there is only one quantity in the pandas.DataFrame')

    return timeseries


# Clouds anomaly
def add_clouds_anomaly(timeseries,sampling_interval,inplace=False):
    if not inplace:
        timeseries = deepcopy(timeseries)

    return add_clouds_effect(timeseries,sampling_interval,inplace=inplace,mv_anomaly=True)


def change_effect_label(timeseries,index,new_effect):
    if not isinstance(new_effect,str):
        raise TypeError('The "new_effect" argument has to be of type string')

    if timeseries.loc[index,'effect_label'] is None:
        timeseries.loc[index,'effect_label'] = new_effect
    elif new_effect in timeseries.loc[index,'effect_label']:
        pass
    else:
        timeseries.loc[index,'effect_label'] += '_' + new_effect

# Noise effect
def add_noise_effect(timeseries,inplace=False):
    if not inplace:
        timeseries = deepcopy(timeseries)
    quantities = _quantities_in(timeseries)

    for quantity in quantities:
        timeseries[quantity] += np.random.normal(0,2,size=len(timeseries))
    for i in range(len(timeseries)):
        change_effect_label(timeseries,i,'noise')
    return timeseries


# Season effect
def calculate_seasonal_sin_value(timeseries,starting_year):
        import calendar
        if calendar.isleap(starting_year):
            seasonal_periodicity = 8784
        else:
            seasonal_periodicity = 8760

        time_offset = dt.datetime(starting_year,2,20,0,0,tzinfo=pytz.UTC)
        seasonal_sin_values = []
        for i in range(len(timeseries)):
            delta_t = timeseries.loc[i,'timestamp'] - time_offset
            time_variable = delta_t.total_seconds()/3600
            sin_value = math.sin((2*math.pi/seasonal_periodicity)*time_variable)
            change_effect_label(timeseries,i,'seasons')
            seasonal_sin_values.append(sin_value)

        return pd.Series(seasonal_sin_values) 


def add_seasons_effect(timeseries,starting_year,inplace=False):

    if not inplace:
        timeseries=deepcopy(timeseries)
    quantities = _quantities_in(timeseries)

    winter_temp = 4.4
    summer_temp = 26
    seasonal_temperature_amplitude = (summer_temp - winter_temp)/2
    winter_humi = 90
    summer_humi = 40
    seasonal_humidity_amplitude = (winter_humi - summer_humi)/2

    if 'temperature' in quantities:
       timeseries['temperature'] += winter_temp + seasonal_temperature_amplitude * calculate_seasonal_sin_value(timeseries,starting_year)
       
    if 'humidity' in quantities:
        timeseries['humidity'] += winter_humi - seasonal_humidity_amplitude * calculate_seasonal_sin_value(timeseries,starting_year)

    return timeseries


# Clouds effect
def add_clouds_effect(timeseries,sampling_interval,inplace=False,mv_anomaly=False):
    if not inplace:
        timeseries=deepcopy(timeseries)
    quantities = _quantities_in(timeseries)

    number_of_points_in_a_day = int(86400/(sampling_interval.total_seconds()))
    # TODO: better handling of this
    if number_of_points_in_a_day == 0:
        raise TypeError('The clouds effect losts significance if the sampling interval is on number of days')

    number_of_days = int(len(timeseries)/number_of_points_in_a_day)
    clouds_factor=0.3
    j = 0

    for i in range(number_of_days):
        is_a_cloudy_day = rnd.randint(0, 1)

        if is_a_cloudy_day:

            for position_in_the_day in range(number_of_points_in_a_day):
                index = i * number_of_points_in_a_day + position_in_the_day

                if position_in_the_day < int(number_of_points_in_a_day/2):
                    clouds_effect_intesity = position_in_the_day/int(number_of_points_in_a_day/2)

                else:
                    clouds_effect_intesity = 2 - (position_in_the_day/int(number_of_points_in_a_day/2))
                change_effect_label(timeseries,index,'clouds')
                for quantity in quantities:

                    if quantity == 'temperature':
                        timeseries.loc[index,'temperature'] *= (1 - (clouds_effect_intesity * clouds_factor))

                    if quantity == 'humidity':

                        if mv_anomaly and j == 0:
                            timeseries.loc[index,'anomaly_label'] = 'clouds'

                        else:
                            timeseries.loc[index,'humidity'] *= (1 + (clouds_effect_intesity * clouds_factor))
            j += 1

    return timeseries


# Spike effect
def add_spike_effect(timeseries,inplace=False,mode='uv'): 
    if not inplace:
        timeseries = deepcopy(timeseries)
    quantities = _quantities_in(timeseries)

    spike_factor = { 'low': 5,
                    'medium': 7,
                    'high': 9
    }
    spike_n = 0
    for i in range(len(timeseries)):
        is_a_spiked_value = True if rnd.randint(0, 50) == 25 else False

        if is_a_spiked_value:
            spike_n += 1
            random_spike_intensity = rnd.choice(list(spike_factor.keys()))
            change_effect_label(timeseries,i,'spike')
            if 'temperature' in quantities:
                timeseries.loc[i,'temperature'] += spike_factor[random_spike_intensity]                  

            if 'humidity' in quantities:

                if (spike_n % 2) == 0:
                    timeseries.loc[i,'humidity'] -= spike_factor[random_spike_intensity]

                else:
                    timeseries.loc[i,'humidity'] += spike_factor[random_spike_intensity]

    return timeseries


def csv_file_maker(timeseries,anomalies=[],effects=[],path=''):
    quantities = _quantities_in(timeseries)
    data_type = ''

    for quantity in quantities:
        data_type += quantity + '_'

    file_name = data_type

    if anomalies:

        for anomaly in anomalies:
            file_name += anomaly + '_'
        file_name += 'anomalies_'

    if effects:
        for effect in effects:
            file_name += effect + '_'
        file_name += 'effects'

    timeseries.to_csv(path + file_name + '.csv', sep=';', index=False, encoding='utf-8')


def plot_func(timeseries,anomalies=[]):
    import matplotlib.pyplot as plt
    quantities = _quantities_in(timeseries)

    colors = { 'temperature': 'crimson',
              'humidity': 'navy'

    }
    anomaly_highlighter = { 'spike_uv': 'red',
                           'spike_mv': 'blue',
                           'step_uv': 'orange',
                           'step_mv': 'yellow',
                           'pattern_uv': 'green',
                           'pattern_mv': 'pink',
                           'noise_uv': 'purple',
                           'noise_mv': 'cyan',
                           'clouds': 'violet'   
    }
    fig, ax = plt.subplots(figsize=(15, 4))

    for quantity in quantities:
        ax.plot(timeseries[quantity],label=quantity,color=colors[quantity])

    ax.set_ylabel(', '.join(quantities))
    start_band_position = timeseries.index[0]
    stop_band_position = timeseries.index[0]

    if anomalies:

        for anomaly in anomalies:
            inside_band = False

            for i in range(len(timeseries)):
                anomaly_target = timeseries.iloc[i]['anomaly_label']

                if anomaly_target == anomaly and not inside_band:
                    start_band_position = timeseries.index[i]
                    inside_band = True

                elif anomaly_target is None and inside_band:
                    stop_band_position = timeseries.index[i]
                    break

                elif anomaly_target == anomaly and inside_band:
                    stop_band_position = timeseries.index[(len(timeseries) - 1)]

                else:
                    continue

            ax.axvspan(start_band_position,stop_band_position,color=anomaly_highlighter[anomaly],alpha=0.3,label=anomaly)
    ax.set_xlabel("timestamp")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


class TimeseriesGenerator:
    pass


class HumiTempTimeseriesGenerator(TimeseriesGenerator):

    def __init__(self, sampling_interval= '15min',
                 time_span='30D',starting_year=None, 
                 starting_month=None, starting_day=None, starting_hour=None, 
                 starting_minute=None,temperature=True, humidity=True):

        self.sampling_interval = pd.Timedelta(sampling_interval).to_pytimedelta()
        self.time_span = pd.Timedelta(time_span).to_pytimedelta()
        self.starting_year = starting_year
        self.starting_month = starting_month
        self.starting_day = starting_day
        self.starting_hour = starting_hour
        self.starting_minute = starting_minute
        self.temperature = temperature
        self.humidity = humidity

    def generate(self,plot=False,generate_csv=False,
                 csv_path='',anomalies=['spike_uv','step_uv'],
                 effects=['noise','seasons','clouds'],index_by_timestamp=True):

        avaliable_anomalies = ['spike_uv','step_uv','noise_uv','pattern_uv','spike_mv','step_mv','noise_mv','pattern_mv','clouds']
        avaliable_effects = ['spike','noise','seasons','clouds']
        for anomaly in anomalies:
            if anomaly not in avaliable_anomalies:
                raise ValueError(f'Anomaly "{anomaly}" is not supported.')
        for effect in effects:
            if effect not in avaliable_effects:
                raise ValueError(f'Effect "{effect}" is not supported.')

        datetime_boundaries = generate_time_boundaries(self.time_span,self.starting_year,
                                                       self.starting_month,self.starting_day,
                                                       self.starting_hour,self.starting_minute)

        timeseries_df = generate_synthetic_humitemp_timeseries(self.sampling_interval,datetime_boundaries,
                                                               self.temperature,self.humidity)
        final_humitemp_timeseries_df = timeseries_df

        if anomalies:

            if isinstance(anomalies,list):

                if 'pattern_uv' in anomalies:
                    final_humitemp_timeseries_df = add_pattern_anomaly(final_humitemp_timeseries_df,
                                                                       self.sampling_interval,mode='uv')
                    if 'pattern_mv' in anomalies:
                        raise ValueError('The injection of anomalies has to be either in univariate mode or in multivariate mode. Cannot select both at the same timestamp')

                if 'pattern_mv' in anomalies:
                    final_humitemp_timeseries_df = add_pattern_anomaly(final_humitemp_timeseries_df,
                                                                       self.sampling_interval,mode='mv')

                if 'spike_uv' in anomalies:
                    final_humitemp_timeseries_df = add_spike_anomaly(final_humitemp_timeseries_df,mode='uv')

                    if 'spike_mv' in anomalies:
                        raise ValueError('The injection of anomalies has to be either in univariate mode or in multivariate mode. Cannot select both at the same timestamp')

                if 'spike_mv' in anomalies:
                    final_humitemp_timeseries_df = add_spike_anomaly(final_humitemp_timeseries_df,mode='mv')

                if 'step_uv' in anomalies:
                    final_humitemp_timeseries_df = add_step_anomaly(final_humitemp_timeseries_df,mode='uv')

                    if 'step_mv' in anomalies:
                        raise ValueError('The injection of anomalies has to be either in univariate mode or in multivariate mode. Cannot select both at the same timestamp')

                if 'step_mv' in anomalies:
                    final_humitemp_timeseries_df = add_step_anomaly(final_humitemp_timeseries_df,mode='mv')

                if 'noise_uv' in anomalies:
                    final_humitemp_timeseries_df = add_anomalous_noise(final_humitemp_timeseries_df,mode='uv')

                    if 'noise_mv' in anomalies:
                        raise ValueError('The injection of anomalies has to be either in univariate mode or in multivariate mode. Cannot select both at the same timestamp')

                if 'noise_mv' in anomalies:
                    final_humitemp_timeseries_df = add_anomalous_noise(final_humitemp_timeseries_df,mode='mv')

                if 'clouds' in anomalies:

                    if 'clouds' not in effects:
                        raise ValueError('Clouds effect must be inside the effects if using clouds anomaly')
                    effects.remove('clouds')
                    final_humitemp_timeseries_df = add_clouds_anomaly(final_humitemp_timeseries_df,
                                                                      self.sampling_interval)

        if effects:

            if isinstance(effects,list):

                if 'noise' in effects:
                    final_humitemp_timeseries_df = add_noise_effect(final_humitemp_timeseries_df)

                if 'seasons' in effects:
                    final_humitemp_timeseries_df = add_seasons_effect(final_humitemp_timeseries_df,
                                                                      datetime_boundaries[0].year)

                if 'clouds' in effects:
                    final_humitemp_timeseries_df = add_clouds_effect(final_humitemp_timeseries_df,
                                                                      self.sampling_interval)

                if 'spike' in effects:
                    final_humitemp_timeseries_df = add_spike_effect(final_humitemp_timeseries_df)

        if index_by_timestamp:
            final_humitemp_timeseries_df.set_index(final_humitemp_timeseries_df['timestamp'],inplace=True)
            final_humitemp_timeseries_df.drop(columns=['timestamp'],inplace=True)

        if plot:
            plot_func(final_humitemp_timeseries_df,anomalies)

        if generate_csv:
            csv_file_maker(final_humitemp_timeseries_df,anomalies,effects,path=csv_path)

        return final_humitemp_timeseries_df


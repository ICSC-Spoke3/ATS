import datetime as dt
import pandas as pd
import random as rnd
import numpy as np
import pytz
import math
from copy import deepcopy

    
def generate_time_boundaries(time_interval='90D', starting_year=None, starting_month= None, starting_day=None, starting_hour=None, starting_minute=None):

    observation_window = pd.Timedelta(time_interval).to_pytimedelta()
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

    #calculating the final day
    final_datetime = starting_datetime + observation_window
    
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
        raise ValueError('Invalid argument: in "time_boundaries", "{}" occurs after "{}". Ensure the time values are in chronological order'.format(time_boundaries[0],time_boundaries[1]))


    temp = []
    humi = []
    time = []
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
        
        anomaly_label.append('normal')
        effect_label.append('bare')
        time.append(time_value)
        
        time_variable = time_value.hour + time_value.minute/60
        sin_value = math.sin((2*math.pi/periodicity)*(time_variable-6))
        
        temp.append(min_temp + temperature_amplitude * (1 + sin_value))
        humi.append(min_humi + humidity_amplitude * (1 - sin_value))
    
        time_value = time_value + delta_t
        if time_value == time_boundaries[1]:
            break


    data_points.update({'time': time})
    data_points.update({'anomaly_label': anomaly_label})
    data_points.update({'effect_label': effect_label})
        
    if temperature:
        data_points.update({'temperature': temp})
    
    if humidity:
        data_points.update({'humidity': humi})
    if not temperature and not humidity:
        raise ValueError('Error: no data selected for creating the DataFrame. Set at True at least one of the two arguments: temperature or humidity')
        
    return pd.DataFrame(data_points)

                                            #anomalies


    

#step_anomaly
def add_step_anomaly(timeseries,mode='uv',inplace=False):
    
    if not inplace:
        timeseries = deepcopy(timeseries)

    quantities = list(timeseries.columns)
    quantities.remove('time')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')
        
    
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
                #print(ramp_factor)
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


#noise_anomaly
def add_anomalous_noise(timeseries,inplace=False,mode='uv'):

    if not inplace:
        timeseries = deepcopy(timeseries)

    quantities = list(timeseries.columns)
    quantities.remove('time')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')
        

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


#pattern_anomaly
def add_pattern_anomaly(timeseries,sampling_interval,inplace=False,mode='uv'):

    if not inplace:
        timeseries = deepcopy(timeseries)

    quantities = list(timeseries.columns)
    quantities.remove('time')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')
        
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
        t0 = timeseries.loc[0,'time']
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


def add_clouds_anomaly(timeseries,sampling_interval,inplace=False):

    if not inplace:
        timeseries = deepcopy(timeseries)

    return add_clouds_effects(timeseries,sampling_interval,inplace=inplace,mv_anomaly=True)

    
    
    
    

    

#effects

def add_noise_effect(timeseries,inplace=False):
 
    if not inplace:
        timeseries = deepcopy(timeseries)

    quantities = list(timeseries.columns)
    quantities.remove('time')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')

    for quantity in quantities:
        
        timeseries[quantity] += np.random.normal(0,2,size=len(timeseries))
    
    return timeseries


def add_seasons_effect(timeseries,starting_year,inplace=False):

    import calendar
    
    if not inplace:
        timeseries=deepcopy(timeseries)

    quantities = list(timeseries.columns)
    quantities.remove('time')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')
    
    winter_temp = 4.4
    summer_temp = 26
    seasonal_temperature_amplitude = (summer_temp - winter_temp)/2
    time_offset = dt.datetime(starting_year,2,20,0,0,tzinfo=pytz.UTC) #start of winter,when temperature is at its minimum 
    
    if calendar.isleap(time_offset.year):
        
        seasonal_periodicity = 8784
    else:
        seasonal_periodicity = 8760

    def insert_seasonal_trend(quantity):
        for i in range(len(timeseries)):

            delta_t = timeseries.loc[i,'time'] - time_offset
            time_variable = delta_t.total_seconds()/3600
            sin_value = math.sin((2*math.pi/seasonal_periodicity)*time_variable)
            seasonal_temperature_trend = winter_temp + seasonal_temperature_amplitude * sin_value
            seasonal_humidity_trend = winter_temp - seasonal_temperature_amplitude * sin_value
        
            timeseries.loc[i,quantity] += seasonal_temperature_trend


    if 'temperature' in quantities and 'humidity' in quantities:
        
        insert_seasonal_trend('temperature')
        insert_seasonal_trend('humidity')

    else:
        insert_seasonal_trend(quantities[0])
        

    return timeseries
        

def add_clouds_effects(timeseries,sampling_interval,inplace=False,mv_anomaly=False):

    if not inplace:
        timeseries=deepcopy(timeseries)

    quantities = list(timeseries.columns)
    quantities.remove('time')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')
        
    number_of_points_in_a_day = int(86400/(sampling_interval.total_seconds()))
    if number_of_points_in_a_day == 0:
        raise TypeError('The clouds effect losts significance if the sampling interval is on number of days')
        #better handling of this
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

                for quantity in quantities:
                    if quantity == 'temperature':
                        timeseries.loc[index,'temperature'] *= (1 - (clouds_effect_intesity * clouds_factor))

                    if quantity == 'humidity':
                        if mv_anomaly and j == 0:
                            #timeseries.loc[index,'humidity'] *= (1 - (clouds_effect_intesity * clouds_factor))
                            timeseries.loc[index,'anomaly_label'] = 'clouds'
                            
                        else:
                            timeseries.loc[index,'humidity'] *= (1 + (clouds_effect_intesity * clouds_factor))
            j += 1
                        

                
    return timeseries    


def add_spike_effect(timeseries,inplace=False,anomaly=False, mode='uv'):
    
    #TODO: change the way anomalous and normal spikes are added     
    if not inplace:
        timeseries = deepcopy(timeseries)

    quantities = list(timeseries.columns)
    quantities.remove('time')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')
    
    spike_factor = { 'low': 5,
                    'medium': 7,
                    'high': 9
    }
    number_of_spikes = 0
    for i in range(len(timeseries)):
        is_a_spiked_value = True if rnd.randint(0, 50) == 25 else False
        
        if is_a_spiked_value:
            number_of_spikes += 1
            random_spike_intensity = rnd.choice(list(spike_factor.keys()))
            
            if not anomaly:
                if 'temperature' in quantities:
                    timeseries.loc[i,'temperature'] += spike_factor[random_spike_intensity]                    
                
                if 'humidity' in quantities:
                    if (number_of_spikes % 2) == 0:
                        timeseries.loc[i,'humidity'] -= spike_factor[random_spike_intensity]
                    else:
                        timeseries.loc[i,'humidity'] += spike_factor[random_spike_intensity]

            if anomaly and number_of_spikes == 10:
                anomaly = False
                if 'temperature' in quantities and 'humidity' in quantities:
                    if mode == 'uv':
                        timeseries.loc[i,'temperature'] -= spike_factor[random_spike_intensity]
                        timeseries.loc[i,'humidity'] += spike_factor[random_spike_intensity]
                        timeseries.loc[i,'anomaly_label'] = 'spike_uv'
                        
                    if mode == 'mv':
                        timeseries.loc[i,'temperature'] -= spike_factor[random_spike_intensity]
                        timeseries.loc[i,'humidity'] -= spike_factor[random_spike_intensity]
                        timeseries.loc[i,'anomaly_label'] = 'spike_mv'
                        

                        
                else:
                    if mode == 'uv':
                        if quantities[0] == 'temperature':
                            timeseries.loc[i,'temperature'] -= spike_factor[random_spike_intensity]
                            timeseries.loc[i,'anomaly_label'] = 'spike_uv'
                            
                        if quantities[0] == 'humidity':
                            timeseries.loc[i,'humidity'] += spike_factor[random_spike_intensity]
                            timeseries.loc[i,'anomaly_label'] = 'spike_uv'

                    if mode == 'mv':
                        raise ValueError(f'Multivariate anomalies cannot be added if ther is only one variable in the timeseries')
    
    return timeseries
    



def csv_file_maker(timeseries,anomalies=[],effects=[],path=''):

    quantities = list(timeseries.columns)
    quantities.remove('time')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')
    
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
    
    quantities = list(timeseries.columns)
    quantities.remove('time')
    quantities.remove('anomaly_label')
    quantities.remove('effect_label')

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
        ax.plot(timeseries['time'],timeseries[quantity],label=quantity,color=colors[quantity])
        
    ax.set_ylabel(', '.join(quantities))


    start_band_position = timeseries.loc[0,'time']
    stop_band_position = timeseries.loc[0,'time']
    if anomalies:
        for anomaly in anomalies:

            inside_band = False
            for i in range(len(timeseries)):
                anomaly_target = timeseries.loc[i,'anomaly_label']
                
                if anomaly_target == anomaly and not inside_band:
                    start_band_position =  timeseries.loc[i,'time']
                    inside_band = True
                   
                elif anomaly_target == 'normal' and inside_band:
                    stop_band_position = timeseries.loc[i,'time']
                    break

                elif anomaly_target == anomaly and inside_band:
                    stop_band_position = timeseries.loc[(len(timeseries) - 1),'time']
                    
                else:
                    continue
                    
            
            ax.axvspan(start_band_position,stop_band_position,color=anomaly_highlighter[anomaly],alpha=0.3,label=anomaly)
            
        
    ax.set_xlabel("time")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

        

class SynteticTimeseriesGenerator:
    pass 






class SyntheticHumiTempTimeseriesGenerator(SynteticTimeseriesGenerator):
    
    def __init__(self, sampling_interval= '15min', observation_window='30D',starting_year=None, starting_month=None, starting_day=None, starting_hour=None, starting_minute=None,temperature=True, humidity=True):
        
        self.sampling_interval = pd.Timedelta(sampling_interval).to_pytimedelta()
        self.observation_window = pd.Timedelta(observation_window).to_pytimedelta()
        self.starting_year = starting_year
        self.starting_month = starting_month
        self.starting_day = starting_day
        self.starting_hour = starting_hour
        self.starting_minute = starting_minute
        self.temperature = temperature
        self.humidity = humidity
        
        

    def generate(self,plot=False,generate_csv=False,csv_path='',anomalies=['spike_uv','step_uv'],effects=['noise','seasons','clouds']):
        
        datetime_boundaries = generate_time_boundaries(self.observation_window, self.starting_year, self.starting_month, self.starting_day, self.starting_hour, self.starting_minute)
        timeseries_df = generate_synthetic_humitemp_timeseries(self.sampling_interval,datetime_boundaries,self.temperature,self.humidity)
        
        final_humitemp_timeseries_df = timeseries_df

        
    
        if anomalies:
            if isinstance(anomalies,list):

                if 'pattern_uv' in anomalies:
                    final_humitemp_timeseries_df = add_pattern_anomaly(final_humitemp_timeseries_df,self.sampling_interval,mode='uv')
                     
                    if 'pattern_mv' in anomalies:
                        raise ValueError('The injection of anomalies has to be either in univariate mode or in multivariate mode. Cannot select both at the same time')

                if 'pattern_mv' in anomalies:
                    final_humitemp_timeseries_df = add_pattern_anomaly(final_humitemp_timeseries_df,self.sampling_interval,mode='mv')
                
                if 'spike_uv' in anomalies:
                    final_humitemp_timeseries_df = add_spike_effect(final_humitemp_timeseries_df,anomaly=True, mode='uv')
                    
                    if 'spike_mv' in anomalies:
                        raise ValueError('The injection of anomalies has to be either in univariate mode or in multivariate mode. Cannot select both at the same time')

                if 'spike_mv' in anomalies:
                    final_humitemp_timeseries_df = add_spike_effect(final_humitemp_timeseries_df,anomaly=True, mode='mv')
                     
                     
                if 'step_uv' in anomalies:
                    final_humitemp_timeseries_df = add_step_anomaly(final_humitemp_timeseries_df,mode='uv')
                     
                    if 'step_mv' in anomalies:
                        raise ValueError('The injection of anomalies has to be either in univariate mode or in multivariate mode. Cannot select both at the same time')

                if 'step_mv' in anomalies:
                    final_humitemp_timeseries_df = add_step_anomaly(final_humitemp_timeseries_df,mode='mv')
                     

                if 'noise_uv' in anomalies:
                    final_humitemp_timeseries_df = add_anomalous_noise(final_humitemp_timeseries_df,mode='uv')
                    
                    if 'noise_mv' in anomalies:
                        raise ValueError('The injection of anomalies has to be either in univariate mode or in multivariate mode. Cannot select both at the same time')

                if 'noise_mv' in anomalies:
                    final_humitemp_timeseries_df = add_anomalous_noise(final_humitemp_timeseries_df,mode='mv')
                    
                    

                if 'clouds' in anomalies:
                    if 'clouds' not in effects:
                        raise ValueError('Clouds effect must be inside the effects if using clouds anomaly')
                    effects.remove('clouds')
                    final_humitemp_timeseries_df = add_clouds_anomaly(final_humitemp_timeseries_df,self.sampling_interval)




        if effects:
            if isinstance(effects,list):
                if 'noise' in effects:
                    final_humitemp_timeseries_df = add_noise_effect(final_humitemp_timeseries_df)
                    
                
                if 'seasons' in effects:
                    final_humitemp_timeseries_df = add_seasons_effect(final_humitemp_timeseries_df,datetime_boundaries[0].year)
                    

                if 'clouds' in effects:
                    final_humitemp_timeseries_df = add_clouds_effects(final_humitemp_timeseries_df,self.sampling_interval)

                if 'spike' in effects:
                    final_humitemp_timeseries_df = add_spike_effect(final_humitemp_timeseries_df,anomaly=False)
                    
                     
                    

        if plot:
        
            plot_func(final_humitemp_timeseries_df,anomalies)

        if generate_csv:
            
            csv_file_maker(final_humitemp_timeseries_df,anomalies,effects,path=csv_path)

        return final_humitemp_timeseries_df



from base import AnomalyDetector

class TimeseriaAnomalyDetector(AnomalyDetector):  OPPURE  MyAnomalyDetector()
    def apply(self,data):
        if not isinstance(data,pd.DataFrame):
            raise NotImplementedError()
        timeseries_df = data
        print(timeseries_df)
        # timeseria format conversion
        model = PeriodicAverageAnomalyDetector()
        model.fit(timeseries)
        timeseries = model.apply(timeseries)
        # convert back
        
    # Implementare fit e apply
    # Aggiungere value_1_anomaly value_2_anomaly anomaly 
    # anomaly = unione (?)
    #timeseries --> df
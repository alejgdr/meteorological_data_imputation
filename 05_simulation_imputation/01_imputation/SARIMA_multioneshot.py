import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from time import time
from dateutil.parser import parse

def exporta(archivo,imputed_column,predi,istep,rango,nombres,path_exported_file='exported_data.csv',save=False):        
     esoru=pd.read_csv(archivo)
     esoru[imputed_column].iloc[istep:istep+rango]=predi[:rango] #agregar nueva columna 
     esoru.time=pd.to_datetime(esoru.time,format='%Y-%m-%d %H:%M:%S')
     esoru.set_index('time',inplace=True)
     if (save==True):
        esoru.to_csv(path_exported_file)
     return(esoru)

print('imported libraries')
fecha1=parse('2018-01-06')-pd.Timedelta('5D')
fecha2=parse('2018-01-06')
isteps=144*5
print(fecha1)
print(fecha2)
tmx_inc=pd.read_csv('../../01_weather_data/03_imputing_process/01_test_data/base_temixco.csv',index_col=0,parse_dates=True)
path_imputed_file='../../01_weather_data/03_imputing_process/02_imputed_data/Tmx_SARIMA_0_1_1_multioneshot.csv'
tmx_inc.to_csv(path_imputed_file)
print('imported files starting imputation cycle')
for day in range(361):
    train_data=tmx_inc.Ig.loc[fecha1:fecha2]
    train_data = train_data.asfreq(pd.infer_freq(train_data.index))
    my_order = (0,0,0)
    my_seasonal_order =(0, 1, 1, 144) #(2, 0, 1, 144)
    model = SARIMAX(train_data, order=my_order, seasonal_order=my_seasonal_order)
    start = time()
    model_fit = model.fit()
    end = time()
    print('training_time:', end-start)
    print('in day:',day)
    predictions = model_fit.forecast(steps=144) #Hago predicci'on
    predictions = np.array(predictions)
    nombres=['time','Ib','Ig','To','RH','WS','WD','P','Eg']
    #Lo meto en el archivo con los indices correspondientes
    imputed=exporta(path_imputed_file,'Ig',predictions,isteps,144,
                        nombres,path_exported_file=path_imputed_file,save=True)
    isteps=isteps+144
    fecha1=fecha1+pd.Timedelta('1D')
    fecha2=fecha2+pd.Timedelta('1D')
    print('fecha1:',fecha1)
    print('imputed fecha2:',fecha2)

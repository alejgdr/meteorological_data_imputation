import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from pickle import load, dump

# def maxmin_season(dailydf,fecha1,fecha2): #identifies the days with the biggest and the lowest Eg on a dataframe
#     df=dailydf.loc[fecha1:fecha2]
#     dfmax=df.Eg.idxmax()
#     dfmin=df.Eg.idxmin()
#     print('dia_maximo:',dfmax)
#     print('dia_minimo:',dfmin) #without columns
def maxmin_season(dailydf,column,fecha1,fecha2):
    df=dailydf.loc[fecha1:fecha2]
    dfmax=df[column].idxmax()
    dfmin=df[column].idxmin()
    print('dia_maximo:',dfmax)
    print('dia_minimo:',dfmin)
    
def deleting_days(df, day1): #converts day1 in dataframe with datetime index into nans
    df[day1]=np.nan
    return (df)

def void_identifier(df,column): #Finds the ubication of data voids on dataframes with no index
    nantmx=df[df[column].isnull()]
    ind=list(nantmx.index)
    isteps=[]
    datavoid=1
    datavoids=[]
    idates=[]
    print('new_void_at:',nantmx.time[ind[0]])
    print('at index',ind[0])
    isteps.append(ind[0])
    idates.append(nantmx.time[ind[0]])
    for x in range(0,len(ind)-1,1):
        if ind[x+1]-ind[x]>1:
            print(datavoid)
            datavoids.append(datavoid)
            datavoid=1
            print('new_void_at:',nantmx.time[ind[x+1]])
            isteps.append(ind[x+1])
            idates.append(nantmx.time[ind[x+1]])
            print('at index',ind[x+1])
        else:
            datavoid+=1
    datavoids.append(datavoid)
    return(isteps,datavoids,idates)
def exporta(archivo,imputed_column,predi,istep,rango,nombres,path_exported_file='exported_data.csv',save=False):        
     esoru=pd.read_csv(archivo)
     esoru[imputed_column].iloc[istep:istep+rango]=predi[:rango] #agregar nueva columna 
     esoru.time=pd.to_datetime(esoru.time,format='%Y-%m-%d %H:%M:%S')
     esoru.set_index('time',inplace=True)
     if (save==True):
        esoru.to_csv(path_exported_file)
     return(esoru)
def seasonal_exporta(archivo,imputed_column,predi,istep,rango,nombres,sol_data_correction=False,save=False,archivo_nombre='imputados_corregidos.csv'): #Sustituye datos de entrada por datos predecidos
     esoru=pd.read_csv(archivo,names=nombres,skiprows=1)
     diff=len(predi)-rango
     esoru.Ig.iloc[istep:istep+rango]=predi.copy()[:-diff] #agregar nueva columna 
     esoru.time=pd.to_datetime(esoru.time,format='%Y-%m-%d %H:%M:%S')
     esoru.set_index('time',inplace=True)
     if (sol_data_correction==True):
        esoru=nightzero(esoru,archivo_nombre,save)
     return(esoru)

def seasonal_pre_process(in_size,out_size,esoru,scalerx,scalery,inputs,outputs,training_step,season_size): #when scaler is already fitted
     set_size=in_size+out_size
     esona=esoru.interpolate(method='polynomial',order=1)
     outna=esona[outputs]
     inpna=esona[inputs]
     train_val_ratio=1
     train_ratio=1
     
     arresoru=np.array(outna)
     arresoruin=np.array(inpna)
     pre_array=[]
     pre_arrax=[]
     arresoruin=scalerx.transform(arresoruin)
     arresoru=scalery.transform(arresoru)
     for set_step in range (0,len(arresoruin)-season_size-out_size,training_step):
         x1=arresoruin[set_step:set_step+in_size]
         pre_arrax.append(x1)
         y=arresoru[set_step+season_size:set_step+season_size+out_size]
         pre_array.append(y)
     y_array,x_array=np.stack(pre_array),np.stack(pre_arrax)
     return (x_array,y_array)

def Multioneshot(esoru,forward_steps,out_size,in_size,istep,model,inputs,outputs,training_step,season_size,scalerx,scalery):
    x_array,y_array=seasonal_pre_process(in_size,out_size,esoru,scalerx,scalery,inputs,outputs,training_step,season_size)
#     pre_process(in_size,out_size,inputs,esoru,scaler,scaler2)
    forward_steps=int(forward_steps/6)*6+6
    output=[]
    target=[]
    for step in range (istep,istep+forward_steps,out_size):
        pry=model.predict(x_array[step].reshape(1,in_size,x_array.shape[2]))
        pry=scalery.inverse_transform(pry)
        tary=y_array[step]
        tary=scalery.inverse_transform(tary)
        output.append(pry)
        target.append(tary)
    predi=np.asarray(output,dtype='object').reshape(forward_steps)
    target=np.asarray(target,dtype='object').reshape(forward_steps)
#     output=np.asarray(output,dtype='object').reshape(forward_steps,1)
#     target=np.asarray(target,dtype='object').reshape(forward_steps,1)
#     output=output.reshape(forward_steps)
    return(predi,target) 

def importa(archivo,nombres):
 esoru=pd.read_csv(archivo,names=nombres,skiprows=1)
 esoru.time=pd.to_datetime(esoru.time,format='%Y-%m-%d %H:%M:%S')
 esoru.set_index('time',inplace=True)
 return(esoru)

def nightzero(df,archivo_nombre,save=True):
    df['diajuliano']=df.index.dayofyear
    df['minutodia']=(df.index.hour*60)+df.index.minute
    df.loc[df.alturasolar<0,'Ig']=0
    df.loc[df.alturasolar<0,'Ib']=0
    dfcorr_noche=df[['Ib','Ig','to','RH','P']]
    if save==True:
        dfcorr_noche.to_csv(archivo_nombre)
    return(dfcorr_noche)
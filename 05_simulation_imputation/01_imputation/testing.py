import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
from pickle import load, dump


def importa(archivo,nombres):
 esoru=pd.read_csv(archivo,names=nombres,skiprows=1)
 esoru.tiempo=pd.to_datetime(esoru.tiempo,format='%Y-%m-%d %H:%M:%S')
 esoru.set_index('tiempo',inplace=True)
 return(esoru)

def seasonal_exporta(archivo,predi,istep,in_size,rango,season_size,nombres,sol_data_correction=False,save=False,archivo_nombre='imputados_corregidos.csv'): #Sustituye datos de entrada por datos predecidos
     esoru=pd.read_csv(archivo,names=nombres,skiprows=1)
     esoru.Global.iloc[istep+season_size:istep+season_size+rango]=predi.copy() #agregar nueva columna 
     esoru.tiempo=pd.to_datetime(esoru.tiempo,format='%Y-%m-%d %H:%M:%S')
     esoru.set_index('tiempo',inplace=True)
     if (sol_data_correction==True):
        esoru=nightzero(esoru,archivo_nombre,save)
     return(esoru)

def seasonal_pre_process(in_size,out_size,esoru,scalerx,scalery,inputs,outputs,training_step,season_size): #when scaler is already fitted
     set_size=in_size+out_size
     esona=esoru.interpolate(method='polynomial',order=1)
     fecha1='2019-01-01'
     fecha2='2019-01-31'
     outna=esona[outputs]#[fecha1:fecha2]
     inpna=esona[inputs]#[fecha1:fecha2]
     train_val_ratio=1#.9 #Qué porcentaje serán los datos de entrenamiento y validación 
     train_ratio=1#.8 #Qué porcentaje serán los datos de solo entrenamiento 
     
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
    output=[]
    target=[]
    for step in range (istep,istep+forward_steps,out_size):
        pry=model.predict(x_array[step].reshape(1,in_size,6))
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

def alturaTMX(N,tiest):
    #(día juliano, tiempo estándar)
    lat=18.8397315*np.pi/180
    logloc=99.2364961
    logest=90
    #N=int(N)
    delta=23.45*np.pi/180*np.sin(((2*np.pi)/365)*(284+N))
    B=(N-1)*((2*np.pi)/365) 
    Et=229.2*(.000075+(.001868*np.cos(B))-(.032077*np.sin(B))-(.014615*np.cos(2*B))-(.04089*np.sin(2*B)))
    cenit=[]
    tsol=tiest+(4*(logest-logloc))+Et
    omega=.25*(tsol-720)
    theta=np.rad2deg(np.arccos((np.cos(lat)*np.cos(np.radians(omega))*np.cos(delta))+(np.sin(delta)*np.sin(lat))))
    altura=90-theta
    return(altura)

def nightzero(df,archivo_nombre,save=True):
    df['diajuliano']=df.index.dayofyear
    df['minutodia']=(df.index.hour*60)+df.index.minute
    df['alturasolar']=alturaTMX(df.diajuliano,df.minutodia)
    df.loc[df.alturasolar<0,'Global']=0
    df.loc[df.alturasolar<0,'Direct']=0
    df.loc[df.alturasolar<0,'Difusa']=0 
    dfcorr_noche=df[['Direct','Global','Difusa','Temperatura','Humedad','Presion']]
    if save==True:
        dfcorr_noche.to_csv(archivo_nombre)
    return(dfcorr_noche)

def dfmetricas(dfimp,impesoru_target,model_name):
    dfrad=impesoru_target.copy()
    dfrad['prediccion']=dfimp.Global.copy().astype(float)
    dfrad['minutodia']=(dfrad.index.hour*60)+dfrad.index.minute
    dfrad['me']=(dfrad.prediccion-dfrad.Global).astype(float)
    dfrad['mae']=np.abs(dfrad.Global-dfrad.prediccion).astype(float)
    dfmingroup=dfrad.groupby(['minutodia',pd.Grouper(freq='1H')]).mean()
    dfmindia=dfmingroup.groupby(pd.Grouper(level='minutodia',axis=0)).mean()
    dfmindia.loc[dfmindia.alturasolar<0,'mae']=np.nan
    meandiay=dfmindia.mae.mean()
    
    dfsamp2=dfrad.resample('D').sum() #this dataframe is used to get the difference of energy 
    dfsamp=dfrad.resample('D').mean()
    
    dfsamp['E_d']=dfsamp2['Global']/6
    dfsamp['Ep_d']=dfsamp2['prediccion']/6
    dfsamp['E_dmape']=(np.abs(dfsamp['E_d']-dfsamp['Ep_d'])/dfsamp['E_d'])*100
    dfsamp['E_dmae']=np.abs(dfsamp['E_d']-dfsamp['Ep_d'])

    dfsamp3=dfsamp.resample('Y').mean()
    dfsamp3['mae_Ig']=meandiay
    dfsamp3['model']=model_name
    return(dfsamp3,dfsamp,dfrad)

def begin_table(infodf,cols_gen,path,nombre_archivo,save=True): #creates a new empty archive to store a dataframe, just use it once
    df=infodf
    if save==True:
        df.to_csv(path+nombre_archivo)
    return(df)

def actualizar_bitacora(infodf,cols_gen,path,nombre_archivo,save=True): #adds a new row on a predetermined dataframe 
    df=pd.read_csv(path+nombre_archivo)
    newdf=pd.concat([df,infodf])
    newdf=newdf.set_index('model')
    if save==True:
        newdf.to_csv(path+nombre_archivo)
    return(newdf)
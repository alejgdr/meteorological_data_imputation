import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def SsTMX(N,tiest):
    #(día juliano, tiempo estándar)
    lat=18.8397315*np.pi/180
    logloc=99.2364961
    logest=90
    #N=int(N)
    delta=23.45*np.pi/180*np.sin(((2*np.pi)/365)*(284+N))
    B=(N-1)*((2*np.pi)/365) 
    Et=229.2*(.000075+(.001868*np.cos(B))-(.032077*np.sin(B))-(.014615*np.cos(2*B))-(.04089*np.sin(2*B)))
    #cenit=[]
    tsol=tiest+(4*(logest-logloc))+Et
    omega=.25*(tsol-720)
    theta=np.rad2deg(np.arccos((np.cos(lat)*np.cos(np.radians(omega))*np.cos(delta))+(np.sin(delta)*np.sin(lat))))
    Ss=(np.sin(np.radians(omega))*np.cos(delta))/np.sin(np.radians(theta))
    #altura=90-theta
    return(Ss)

def CsTMX(N,tiest):
    #(día juliano, tiempo estándar)
    lat=18.8397315*np.pi/180
    logloc=99.2364961
    logest=90
    #N=int(N)
    delta=23.45*np.pi/180*np.sin(((2*np.pi)/365)*(284+N))
    B=(N-1)*((2*np.pi)/365) 
    Et=229.2*(.000075+(.001868*np.cos(B))-(.032077*np.sin(B))-(.014615*np.cos(2*B))-(.04089*np.sin(2*B)))
    #cenit=[]
    tsol=tiest+(4*(logest-logloc))+Et
    omega=.25*(tsol-720)
    theta=np.rad2deg(np.arccos((np.cos(lat)*np.cos(np.radians(omega))*np.cos(delta))+(np.sin(delta)*np.sin(lat))))
    Cs=((np.sin(lat)*np.cos(np.radians(omega))*np.cos(delta))-(np.cos(np.radians(theta))*np.sin(delta)))/np.sin(np.radians(theta))
    #altura=90-theta
    return(Cs)

def gammaprimaTMX(N,tiest):
    #(día juliano, tiempo estándar)
    lat=18.8397315*np.pi/180
    logloc=99.2364961
    logest=90
    #N=int(N)
    delta=23.45*np.pi/180*np.sin(((2*np.pi)/365)*(284+N))
    B=(N-1)*((2*np.pi)/365) 
    Et=229.2*(.000075+(.001868*np.cos(B))-(.032077*np.sin(B))-(.014615*np.cos(2*B))-(.04089*np.sin(2*B)))
    #cenit=[]
    tsol=tiest+(4*(logest-logloc))+Et
    omega=.25*(tsol-720)
    theta=np.rad2deg(np.arccos((np.cos(lat)*np.cos(np.radians(omega))*np.cos(delta))+(np.sin(delta)*np.sin(lat))))
    gammaprima=np.rad2deg(np.arctan((np.sin(np.radians(omega))*np.cos(delta))/((np.sin(lat)*np.cos(np.radians(omega))*np.cos(delta))-(np.cos(np.radians(theta))*np.sin(delta)))))
    #altura=90-theta
    return(gammaprima)

def azimuthTMX(N,tiest):
    if (CsTMX(N,tiest)>=0).bool:
        azimuth=gammaprimaTMX(N,tiest)
    if (CsTMX(N,tiest)<0 & SsTMX(N,tiest)<0).bool:
        azimuth=-180+gammaprimaTMX(N,tiest)
    if (CsTMX(N,tiest)<0 & SsTMX(N,tiest)>0).bool:
        azimuth=180+gammaprimaTMX(N,tiest)
    return (azimuth)

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

def nightzero_timeprep(df,saved_file='saved_file.csv',save=True,saving_path='../01_weather_data/02_cleaned_data/'):
#adds solar position data (azimuth and solar altitude), forces night solar values to zero and saves a file with these values
    df['diajuliano']=df.index.dayofyear
    df['minutodia']=(df.index.hour*60)+df.index.minute
    df['alturasolar']=alturaTMX(df.diajuliano,df.minutodia)
    df['Cs']=CsTMX(df.diajuliano,df.minutodia)
    df['Ss']=SsTMX(df.diajuliano,df.minutodia)
    df['azimuth']=gammaprimaTMX(df.diajuliano,df.minutodia)
    df.loc[(df.Cs<0) & (df.Ss<0),'azimuth']=df.loc[(df.Cs<0) & (df.Ss<0),'azimuth']-180
    df.loc[(df.Cs<0) & (df.Ss>0),'azimuth']=df.loc[(df.Cs<0) & (df.Ss>0),'azimuth']+180
    df.loc[df.alturasolar<0,'prediccion']=0
    df.loc[df.alturasolar<0,'Ig']=0
    df.loc[df.alturasolar<0,'Ib']=0
    dfcorr_noche=df[['Ib','Ig','to','RH','P','WS','WD','alturasolar','azimuth']]
    if save==True:
        dfcorr_noche.to_csv(saving_path+saved_file)
    return(dfcorr_noche)
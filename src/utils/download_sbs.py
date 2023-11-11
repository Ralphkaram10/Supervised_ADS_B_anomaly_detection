"""
"download_sbs.py" is a python script used to download ADSB raw data from the opensky
network and convert them to SBS files (each SBS file is associated with one
callsign between a first seen datetime and a last seen datetime). The
associated callsigns are chosen at random between a chosen start datetime and
stop datetime. To be able to access the opensky network to download SBS files
with the current python script, a username and password of a valid opensky
network account should be added to the configuration file (see:
https://traffic-viz.github.io/opensky_impala.html). Note: A directory
"SBS_dataset" will be created in the 'datasets' directory of this repository 
and it will contain the downloaded SBS files (or they will be added to the 
directory if it already exists). 

NEEDED PACKAGES

    pandas (could be installed with conda or pip),
    tqdm (could be installed with conda or pip),
    traffic (could be installed with pip).


USAGE

    usage: download_sbs.py [-h] --start START --stop STOP --nb_callsigns
			   NB_CALLSIGNS

    optional arguments:
      -h, --help            show this help message and exit
      --start START         Specify the start datetime (format: "yyyy-MM-dd
			    h:m:s")
      --stop STOP           Specify the stop datetime (format: "yyyy-MM-dd h:m:s")
      --nb_callsigns NB_CALLSIGNS
			    Select the number of callsigns' SBS files you want to
			    obtain

"""

import argparse
import math
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
from datetime import timedelta  
import traffic
from traffic.data import opensky
from traffic.data.samples import belevingsvlucht
from pathlib import Path
import numpy as np
import math
import random
from tqdm import tqdm
import scipy.signal
import os
import sys
from scipy import signal,stats
import configparser


def get_list_callsigns(start='2019-10-25 00:00:00.0',stop='2019-10-27 00:00:00.0',limit=10000):
    """
    This function gives a list of random callsigns corresponding to flights between 
    a start time and stop time
    """
    flight_list=opensky.flightlist(start=start,stop=stop,limit=limit)
    flight_list=flight_list[['callsign','firstseen','lastseen']].drop_duplicates()    
    return flight_list.callsign.values
    

def get_raw_data_for_given_callsign(callsign='QXE2355' ,start='2019-10-25 00:00:00.0',stop='2019-10-27 00:00:00.0'):
    """
    This function gives raw data corresponding to a specific callsign
    """
    try:
       flight=opensky.flightlist(callsign=callsign ,start=start ,stop=stop)
       
       f=flight.iloc[0]
       print('callsign:'+str(f.callsign)+' firstseen:'+str(f.firstseen)+' lastseen:'+str(f.lastseen))
       test1=(f.callsign is not None);assert test1,"no callsign";
       test2=isinstance(f.firstseen,pd._libs.tslibs.timestamps.Timestamp);assert test2,"no first seen"
       test3=isinstance(f.lastseen,pd._libs.tslibs.timestamps.Timestamp);assert test3,"no last seen"

       raw_df=opensky.rawdata(start=f.firstseen,stop=f.lastseen,callsign=[f.callsign],date_delta=datetime.timedelta(hours=1),cached=True)#opensky.rawdata(start=f.firstseen,stop=f.lastseen,callsign=[f.callsign],cached=True)
       
       print('nb_messages=',str(len(raw_df.data)))        
       
       raw_decoded_f=raw_df.decode()
       
       assert raw_decoded_f is not None, "Decoding gave None"
       
       raw_decoded_f=raw_decoded_f.data
       
       raw_decoded_f=raw_decoded_f.sort_values(by='timestamp', ascending=True)
       raw_decoded_f=raw_decoded_f.reset_index(drop=True)#necessary for some cases
       
       #print("plot before filtering and before conversion to sbs")
       #fig, ax = plt.subplots()
       #ax.plot(raw_decoded_f['altitude'])
       #plt.show()
 
    except AssertionError as error:
       print(error)
       print("callsign "+str(callsign)+" not able to be downloaded or decoded")   
       raw_decoded_f=None
    return raw_decoded_f



def remove_plus_sign(x):
    """
    This function removes the time zone string from a datetime string
    """
    if ('+00:' in x):
        return x.replace("+00:",".000")
    else:
        return x
        
    
def change_name_of_columns(sbs_df):
    """
    Change the name of raw data columns of a preliminary sbs compatibility
    """
    lookup_dict={'timestamp':'f7f8',
     'icao24':'4',
     'groundspeed':'13',
     'track':'14',
     'vertical_rate':'17',
     'altitude':'12',
     'IAS':'',
     'heading':'',
     'Mach':'',
     'vertical_rate_barometric':'',
     'vertical_rate_inertial':'',
     'callsign':'11',
     'geoaltitude':'',
     'squawk':'18',
     'selected_fms':'',
     'selected_mcp':'',
     'barometric_setting':'',
     'latitude':'15',
     'longitude':'16',
     'onground':'22',
     'roll':'',
     'TAS':'',
     'track_rate':''}
    sbs_df.columns=sbs_df.columns.to_series().map(lookup_dict)
    return sbs_df;


def pre_sbs_compatibility(sbs_df):
    """
    Preprocessing of the raw data to be compatible with the sbs format (nb: the message type is not yet specified)
    """
    print('pre sbs compatibility')
    
    sbs_df['f7f8']=sbs_df.f7f8.apply(str)
    sbs_df[['7','8']] = sbs_df.f7f8.str.split(" ",expand=True,)
    cols=[i for i in ['f7f8',''] if i in sbs_df.columns]
    sbs_df=sbs_df.drop(cols, axis=1)
    cols=[i for i in ['7','8','4','13','14','17','12','11','18','15','16','22'] if i in sbs_df.columns]
    sbs_df=sbs_df[cols]
    

    sbs_df['1']='MSG'
    sbs_df['2']='2'
    sbs_df['3']='3' 
    
    sbs_df['5']=sbs_df['4'].copy()
    sbs_df['4']=sbs_df['4'].apply(lambda x: str(int(str(x),16))[0:6])
    sbs_df['6']=sbs_df['4'].copy()
    
    sbs_df['7']=sbs_df['7'].apply(lambda s:s.replace("-", "/"))
    
    sbs_df['8']=sbs_df['8'].apply(lambda s:s[0:12])
    sbs_df['8']=sbs_df['8'].apply(remove_plus_sign)
    
    
    sbs_df['9']=sbs_df['7'].copy()
    sbs_df['10']=sbs_df['8'].copy()
    
    sbs_df['19']=np.nan
    sbs_df['20']=np.nan
    sbs_df['21']=np.nan
    
    
    right_order=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']
    sbs_df=sbs_df.reindex(right_order, axis=1)
    return sbs_df






def isnan(x):
    if isinstance(x, str):
        return False
    elif x is None:
        return True
    else:
        return math.isnan(float(x))
    
    
def notnan(x):
    return not isnan(x)


def specify_message_type(sbs_message):
    """
    Specifying the sbs message type:
    1 Identification and Category (important)
    2 Surface Position Message
    3 Airborne Position Message(important)           
    4 Airborne Velocity Message(important)
    5 Surveillance Alt Message
    6 Surveillance ID Message
    7 Air To Air Message
    8 All Call Reply
    nb: only the types 1, 2, 3, 4 and 5
    were assigned to the messages in this script
    """
    the_10_first_cells=(
    notnan(sbs_message['1']) and notnan(sbs_message['2']) and notnan(sbs_message['3']) and notnan(sbs_message['4']) and
    notnan(sbs_message['5']) and notnan(sbs_message['6']) and notnan(sbs_message['7']) and notnan(sbs_message['8']) and 
    notnan(sbs_message['9']) and notnan(sbs_message['10'])
    )
    
    test1=(
    the_10_first_cells and notnan(sbs_message['11'])  and isnan(sbs_message['12']) and 
    isnan(sbs_message['13']) and isnan(sbs_message['14']) and isnan(sbs_message['15'])  and isnan(sbs_message['16']) and 
    isnan(sbs_message['17']) and isnan(sbs_message['18']) and isnan(sbs_message['19'])  and isnan(sbs_message['20']) and
    isnan(sbs_message['21']) and isnan(sbs_message['22'])
    )
    
    
    test3=(
    notnan(the_10_first_cells) and notnan(sbs_message['11'])  and notnan(sbs_message['12']) and 
    isnan(sbs_message['13']) and isnan(sbs_message['14']) and notnan(sbs_message['15'])  and notnan(sbs_message['16']) and 
    isnan(sbs_message['17']) and isnan(sbs_message['18']) and isnan(sbs_message['19'])  and isnan(sbs_message['20']) and
    isnan(sbs_message['21']) and notnan(sbs_message['22'])  
    )
    
    test4=(
    the_10_first_cells and notnan(sbs_message['11'])  and isnan(sbs_message['12']) and 
    notnan(sbs_message['13']) and notnan(sbs_message['14']) and isnan(sbs_message['15'])  and isnan(sbs_message['16']) and 
    notnan(sbs_message['17']) and isnan(sbs_message['18']) and isnan(sbs_message['19'])  and isnan(sbs_message['20']) and
    isnan(sbs_message['21']) and isnan(sbs_message['22'])
    )
    
    test5=(
    the_10_first_cells and notnan(sbs_message['11'])  and notnan(sbs_message['12']) and 
    isnan(sbs_message['13']) and isnan(sbs_message['14']) and isnan(sbs_message['15'])  and isnan(sbs_message['16']) and 
    isnan(sbs_message['17']) and isnan(sbs_message['18']) and isnan(sbs_message['19'])  and isnan(sbs_message['20']) and
    isnan(sbs_message['21']) and isnan(sbs_message['22'])
    )
    
    if test1:
       sbs_message['2']='1'
                        
    elif test3:
       sbs_message['2']='3' 
       sbs_message['11']=np.nan
       sbs_message['19']='0'   
       sbs_message['20']='0'
       sbs_message['21']='0'
       if sbs_message['22']==True:
           sbs_message['22']='1'
       elif sbs_message['22']==False:
           sbs_message['22']='0'
       else:
           sbs_message['22']='0'
       
    elif test4:
        sbs_message['2']='4'
        sbs_message['11']=np.nan
        
    elif test5:
        sbs_message['2']='5'
        sbs_message['19']='0'
        sbs_message['21']='0'
        sbs_message['22']='0'
        
    else:
        sbs_message=np.nan
    
    return sbs_message


def remove_excess_cells(sbs_message):
    """
    Remove the excess cells in sbs messages
    """
    print('remove excess cells')
    
    if sbs_message['2']=='1':
        sbs_message[['12','13','14','15','16','17','18','19','20','21','22']]=np.nan
    
    elif sbs_message['2']=='3':
        sbs_message[['13','14','17','18','19','20','21']]=np.nan
        
    elif sbs_message['2']=='4':
        sbs_message[['12','15','16','18','19','20','21','22']]=np.nan
        
    elif sbs_message['2']=='5':
        sbs_message[['13','14','15','16','17','18','19','20','21','22']]=np.nan
    print(sbs_message)
    return sbs_message
        


def replace_notnan_with_1(df_row1):
    """
    Utility function used in median filter
    """
    df_row=df_row1.copy()
    for index, value in df_row.items():
        row=df_row[index]
        if notnan(row):
            df_row[index]=1
    return df_row
    
def median_filter(df,targeted_columns,window=3):
    """
    Median filter applied on decoded raw data before their conversion to the sbs format
    """
    if (set(targeted_columns).issubset(set(df.columns))):#to prevent a rare error (lack of features in the downloaded data using the traffic library)
        keep_nans_multiplicator=df.loc[:,targeted_columns].apply(replace_notnan_with_1)    
        df.loc[:,targeted_columns]=df.loc[:,targeted_columns].ffill()
        for col in targeted_columns:
            numpy_filtered_data=signal.medfilt(df.loc[:,col], window)
            df.loc[:,col]=pd.DataFrame(data=np.round(numpy_filtered_data,5), columns=[col])
        df.loc[:,targeted_columns]=keep_nans_multiplicator*df.loc[:,targeted_columns]
    else:
        df=None
    return df



def from_raw_to_sbs(raw_decoded_f,filtering_noise=0):
    """
    Conversion from raw to sbs
    """
    print('from raw_decoded to sbs')
    sbs_df=raw_decoded_f.copy()
    if filtering_noise==1:
        sbs_df=median_filter(sbs_df,['altitude','groundspeed','track','latitude','longitude'],window=7)
    if not sbs_df is None:

        #print("plot after filtering and before conversion to sbs")
        #fig, ax = plt.subplots()
        #ax.plot(sbs_df['altitude'])   
        #plt.show()
           
        sbs_df=change_name_of_columns(sbs_df)
        sbs_df=pre_sbs_compatibility(sbs_df)
        print('treatment of each message for sbs compatibility')
        sbs=sbs_df.iloc[0:-1].apply(specify_message_type,axis=1)    
        sbs=sbs.dropna(how='all')        
        sbs=sbs.reset_index(drop=True)
    else:
           sbs=None
    return sbs;


def create_sbs_files_from_callsign_list(callsign_list=['QXE2355'],start='2019-10-25 00:00:00.0',stop='2019-10-27 00:00:00.0',filtering_noise=0):
    """
    Creating sbs files corresponding to a specified callsign list
    """
    SBS_dataset_path='../../datasets/SBS_dataset'
    if filtering_noise==0:
        last_extension='_sbs.sbs'
    elif filtering_noise==1:
        last_extension='_filtered.sbs'
    sbs_dict={}
    if not os.path.exists(SBS_dataset_path):
       os.makedirs(SBS_dataset_path)
       start_index=0
    else:
        #if you stop downloading by interrupting the script, when you restart the script the downloading will continue from the last downloaded SBS file (this is the only file that will be redownloaded)
        #if you want to restart all the downloading change the following line to "start_index=0"
        start_index=len(os.listdir('SBS_dataset'))-1
        if start_index<0:
            start_index=0
    end_index=len(callsign_list)
    assert end_index>start_index, "The number of required SBS files is smaller then what is already obtained (choose a bigger number i.e. bigger then what is already obtained)"
    for i in tqdm(range(start_index,end_index)):
        print('flight: '+str(i+1)+'/'+str(end_index-start_index))
        callsign=callsign_list[i]
        sbs_file_name=SBS_dataset_path+'/callsign_'+str(callsign)+last_extension
        if os.path.exists(sbs_file_name):#skip existing files
            continue
        raw_decoded_f=get_raw_data_for_given_callsign(callsign=callsign ,start=start,stop=stop)
        if not raw_decoded_f is None:
            sbs=from_raw_to_sbs(raw_decoded_f,filtering_noise=filtering_noise)
            
            if isinstance(sbs, pd.DataFrame):#because we might get series of nones in some rare cases which are inacceptable
                sbs.to_csv(sbs_file_name, sep=',', encoding='utf-8',header=False,index=False)
                sbs_dict[callsign]=sbs
            else:
                print("Could not transform raw data to sbs")
        else:
            print("Could not get raw data for the flight")            
    return sbs_dict


if __name__=='__main__':
    try:
        parser = argparse.ArgumentParser(description="The current python script is used to download ADSB raw data from the opensky network and convert them to SBS files (each SBS file is associated with one callsign between a first seen datetime and a last seen datetime). The associated callsigns are chosen at random between a chosen start datetime and stop datetime. To be able to access the opensky network to download SBS files with the current python script, a username and password of a valid opensky network account should be added to  the configuration file (see: https://traffic-viz.github.io/opensky_impala.html). Note: A directory \"SBS_dataset\" will be created and it will contain the downloaded SBS files (or they will be added to the directory if it already exists). The obtained SBS files can be attacked with the false data injection software (FDIT).") 
        parser.add_argument("--start", help="Specify the start datetime (format: \"yyyy-MM-dd h:m:s\")",required=True,type=str)
        parser.add_argument("--stop", help="Specify the stop datetime (format: \"yyyy-MM-dd h:m:s\")",required=True,type=str)
        parser.add_argument("--nb_callsigns", help="Select the number of callsigns' SBS files you want to obtain",required=True,type=int)
        args = parser.parse_args()

        #connect to opensky
        print(f'configpath={traffic.config_file}\n')
        traffic.config_file
        parser = configparser.ConfigParser()
        parser.read(traffic.config_file)
        username=parser.get("opensky", "username")
        password=parser.get("opensky", "password")
        opensky.username=username
        opensky.password=password

        #apparently you cannot decode old data (more than a year) and you won't get enough raw data either
        date_time_str1= args.start #'2022-10-25 00:00:00.0'
        date_time_str2= args.stop #'2022-10-27 00:00:00.0'
        nb_callsigns= args.nb_callsigns
        
        callsign_list=get_list_callsigns(start=date_time_str1,stop=date_time_str2,limit=nb_callsigns)        
        callsign_list=list(dict.fromkeys(callsign_list))#to remove duplicate callsigns

        callsign_list=callsign_list[0:nb_callsigns]
        sbs_dict=create_sbs_files_from_callsign_list(callsign_list=callsign_list,start=date_time_str1,stop=date_time_str2,filtering_noise=1)
        print('\n\nNote: if you get this last error of the paramiko package (AttributeError: \'NoneType\' object has no attribute \'time\'), it has no effect on the script at all and the files are still downloaded\n\n ')
        
    except KeyboardInterrupt:
        print('Keyboard Interruption')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

            
    
 

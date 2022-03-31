# -*- coding: UTF-8 -*-
####################################################################
__author__ = 'Section Hydrologie et Ecohydraulique--Olivier Champoux@canada.ca'
__version__ = '2021CR3'
__date__ = '2021-12-20'
__update__ = ''
__copyright__ = 'Copyright (c) 2021 Environment Canada'
# *****************************************************************************

####### TESTING STOCASTIC SERIES #####################

import os,sys,time,getopt,shutil
import numpy as np
import pandas as pd
import scipy.interpolate
import logging
import matplotlib.pyplot as plt
from optparse import OptionParser
import configparser
import WBMGrapher
import WBMDynamicPlotter
import WBMDynamicFlowPlotter
import WBMDynamicPlotter_CC
from tqdm import tqdm

def PrintParser(InfuncParser):
    for section_name in InfuncParser.sections():      
        print('Section:', section_name)
        print('  Options:', InfuncParser.options(section_name))

        for name, value in InfuncParser.items(section_name):
            print('  %s = %s' % (name, value))


def FlowInterpolator(current_level,H,Q,extrapolate_log,DATE):
    try:
        f1 = scipy.interpolate.interp1d(H, Q,kind = 'linear')
        InterQ = f1(current_level)
    except ValueError:
        f1 = scipy.interpolate.interp1d(H, Q, kind='linear',fill_value='extrapolate')
        extrapolate_log.info(f'# EXTRAPOLATION OF FLOW-----CURRENT LEVEL AT  {current_level}  ------{DATE}')
        InterQ = f1(current_level)
    return float(InterQ)

def MarinaLevelInterpolator(Virtual_level,H_VIRT,H_SJ,extrapolate_log,DATE):
    try:
        f1 = scipy.interpolate.interp1d(H_VIRT,H_SJ,kind = 'linear')
        InterMarina = f1(Virtual_level)
    except ValueError:
        f1 = scipy.interpolate.interp1d(H_VIRT, H_SJ, kind='linear',fill_value='extrapolate')
        extrapolate_log.info(f'# EXTRAPOLATION OF VIRTUAL LEVEL-----CURRENT LEVEL AT  {Virtual_level}  ------{DATE}')
        InterMarina = f1(Virtual_level)

    return float(InterMarina)


def MarinaLevelInterpolator_delta(Virtual_level,H_VIRT,DELTA,extrapolate_log,DATE):
    try:
        f1 = scipy.interpolate.interp1d(H_VIRT,DELTA,kind = 'linear')
        InterDelta = f1(Virtual_level)
    except ValueError:
        f1 = scipy.interpolate.interp1d(H_VIRT, DELTA, kind='linear',fill_value='extrapolate')
        extrapolate_log.info(f'# EXTRAPOLATION OF VIRTUAL LEVEL-----CURRENT LEVEL AT  {Virtual_level}  ------{DATE}')
        InterDelta = f1(Virtual_level)

    InterMarina=Virtual_level+InterDelta
    return float(InterMarina)




def MarinaLevelInterpolatorFLOW(Current_flow,FLOW,H_SJ,extrapolate_log,DATE):
    #print(Virtual_level)
    try:
        f1 = scipy.interpolate.interp1d(FLOW,H_SJ,kind = 'linear')
        InterMarina = f1(Current_flow)
    except ValueError:
        f1 = scipy.interpolate.interp1d(FLOW, H_SJ, kind='linear',fill_value='extrapolate')
        extrapolate_log.info(f'# EXTRAPOLATION OF VIRTUAL LEVEL-----CURRENT LEVEL AT  {Current_flow}  ------{DATE}')
        InterMarina = f1(Current_flow)
    return float(InterMarina)


def LakeVolumeInterpolator(current_level, Level, Volume):
    f1 = scipy.interpolate.interp1d(Level, Volume, kind='linear', fill_value=(23407195603, 34401717510),
                                    bounds_error=False)
    InterVolume = f1(current_level)
    return float(InterVolume)


def LakeLevleInterpolatorFromVolume(current_Volume, Volume, Level):

    f1 = scipy.interpolate.interp1d(Volume, Level, kind='linear', fill_value=(26, 35), bounds_error=False)
    try:
        InterLevel = f1(current_Volume)
    except ValueError:
        print(current_Volume)
        quit()

    return float(InterLevel)


def CrossSectionAreaInterpolator(current_level, Level, Area):
    # InterArea = np.interp(current_level,Level,Area)
    f1 = scipy.interpolate.interp1d(Level, Area, kind='linear', fill_value=(0.35112385, 15031.40237),
                                    bounds_error=False)
    try:
        InterArea = f1(current_level)
    except ValueError:
        print(current_level)
        quit()
    return float(InterArea)


def WettedPerimeterInterpolator(current_level, Level, WettedPerimeter):
    # InterPerimeter = np.interp(current_level,Level,WettedPerimeter)
    f1 = scipy.interpolate.interp1d(Level, WettedPerimeter, kind='linear', fill_value=(23.95048958, 2849.56),
                                    bounds_error=False)
    InterPerimeter = f1(current_level)
    return float(InterPerimeter)


def Get_Marina_level(H,parameters):
    try:
        H_SJ = np.polyval(parameters,H)
        #print(Q_SJ)

        if np.isnan(H_SJ):
            print('''######################## CRITICAL ERROR ####################\n
        Error in the provided equation parameters check for spaces make sure values are good, whiteline etc .....please see line 11 of configuration file\n
        section [MARINA] [item] equation_parameters:''')
            exit()

    except RuntimeWarning:
        print('''######################## RuntimeWarning CRITICAL ERROR ####################\n
        Error in the provided equation parameters check for spaces make sure values are good, whiteline etc .....please see line 11 of configuration file\n
        section [MARINA] [item] equation_parameters:''')
        exit()

    return float(H_SJ)



def Get_flow_StageDischargeFromLevel(H,parameters):
    #print(H)
    #print(parameters)
    #quit()
    try:
        Q_SJ = np.polyval(parameters,H)
        #print(Q_SJ)

        if np.isnan(Q_SJ):
            print('''######################## CRITICAL ERROR ####################\n
        Error in the provided equation parameters check for spaces make sure values are good, whiteline etc .....please see line 11 of configuration file\n
        section [STAGE-DISCHARGE] [item] equation_parameters:''')
            exit()

    except RuntimeWarning:
        print('''######################## RuntimeWarning CRITICAL ERROR ####################\n
        Error in the provided equation parameters check for spaces make sure values are good, whiteline etc .....please see line 11 of configuration file\n
        section [STAGE-DISCHARGE] [item] equation_parameters:''')
        exit()

    return float(Q_SJ)

def Get_Canal_flow(Q,parameters):
    #print(H)
    #print(parameters)
    #quit()
    try:
        Q_CANAL = np.polyval(parameters,Q)
        #print(Q_SJ)

        if np.isnan(Q_CANAL):
            print('''######################## CRITICAL ERROR ####################\n
        Error in the provided equation parameters check for spaces make sure values are good, whiteline etc .....please see line 11 of configuration file\n
        section [CANAL] [item] equation_parameters:''')
            exit()

    except RuntimeWarning:
        print('''######################## RuntimeWarning CRITICAL ERROR ####################\n
        Error in the provided equation parameters check for spaces make sure values are good, whiteline etc .....please see line 11 of configuration file\n
        section [CANAL] [item] equation_parameters:''')
        exit()

    return float(Q_CANAL)


def Get_WaterLevel_SaintJean_geom(_Level_ST_JEAN,distance_to_section,_Lake_Level,_River_flow,_Manning,_CrossSectionArea,_WettedPerimeter):
    return +(_Level_ST_JEAN+_Lake_Level-distance_to_section*(_River_flow*_Manning/((_CrossSectionArea**(1.66666667))/(_WettedPerimeter**(0.6666666666))))**2)/2.0

def MarinaLevelCorrector(myRow,Slope_df,Alternative):

    if Alternative=='CURRENT':
        Alternative='Baseline'

    if Alternative=='CRUMP':
        Alternative='alt1'

    if Alternative=='CRUMP_DIV':
        Alternative='alt3'

    if myRow.RICH_FLOW_END <300:
        if Alternative=='alt3':
            Alternative='alt1'

        distance = 36866
        Q=myRow.RICH_FLOW_END
        f1 = scipy.interpolate.interp1d(Slope_df[rf'{Alternative}_discharge'],Slope_df[rf'{Alternative}_slope'],kind = 'linear')
        InterMarinaSlope = f1(Q)


        diff_level_correction=distance*InterMarinaSlope


        marina_level=myRow.LAKE_LEVEL-diff_level_correction

        return marina_level

    else:
        return myRow.LEVEL_MARINA




    #except ValueError:
    #    f1 = scipy.interpolate.interp1d(H_VIRT, H_SJ, kind='linear',fill_value='extrapolate')
    #    extrapolate_log.info(f'# EXTRAPOLATION OF VIRTUAL LEVEL-----CURRENT LEVEL AT  {Virtual_level}  ------{DATE}')
    #    InterMarina = f1(Virtual_level)

    #return float(InterMarina)

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was


def CreateLogger(loggerFile):
    # create logger with 'spam_application'
    logger = logging.getLogger('WBM')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(loggerFile)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger



def main(argv):
    cfgFile = ''

    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('tesLCRR_WBM_3.0.0.py -cfgFile <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('LCRR_WBM_3.0.0.py -cfgFile <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            cfgFile = arg
        elif opt in ("-i", "--ifile"):
            cfgFile = arg
    if cfgFile=='':
        print('''######################## WARNING ####################\n
        PLEASE PROVIDE Configuration file name located in the cfg folder\n
        python LCRR_WBM_2.0.0.py -i <my cfg file name>''')
        exit()
    print('cfg file is {}'.format(cfgFile))


    #################### Basepath ############################
    basepath=(os.path.dirname(os.path.realpath(__file__)))
    data_folder=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data')))
    output_folder=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'output')))
    xls_outputs=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'output','Excel_files')))
    cfg_folder=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'cfg')))
    plan_folder=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'plan')))
    log_folder = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'log')))
    graphs_folder = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output', 'html_graphs', )))




    #CREATE AND READ THE PARSER
    parser=configparser.ConfigParser()
    parser.read(os.path.join(cfg_folder,cfgFile))
    PrintParser(parser)

    USER_FOLDER=False
    #CHECH if USER Output folder is specified
    if parser.get('RESULTS','folder') !='None':
        USER_FOLDER=True
        user_output_folder=os.path.join(output_folder,parser.get('RESULTS','folder'))
        #check if folder exists
        if not os.path.isdir(user_output_folder):
            os.makedirs(user_output_folder)




    #################################### SCENARIO_CONFIGURATION (NBS and Stage-Discharge) #####################################
    NBS_supply_scenario=parser.get('NBS_serie','name')
    NBS_type=parser.get('NBS_serie','type')
    NBS_MODEL = parser.get('NBS_serie', 'model')

    Shoal_modification=parser.get('STAGE-DISCHARGE','name')


    # CREATE THE LOGGER

    Log_file=os.path.join(log_folder,''.join('WBM_{}.log'.format(Shoal_modification)))


    if os.path.isfile(Log_file):
        os.remove(Log_file)
    WBM_logger=CreateLogger(Log_file)


    #checkk if the cfgFile exists
    if not os.path.isfile(os.path.join(cfg_folder,cfgFile)):
        #check for full path
        #if not os.path.isfile(os.path.join(cfg_folder, cfgFile)):

        print('''######################## CRITICAL ERROR ####################\n
        The provided configuration file does not exits in the cfg folder''')
        exit()
    else:
        # copy the config file into the output folder

        shutil.copy(os.path.join(cfg_folder, cfgFile), os.path.join(user_output_folder, cfgFile))



    #Check if the baseline file exists
    if not os.path.isfile(os.path.join(xls_outputs,''.join(('Results_',parser.get('GRAPHICS','Baseline'),'{}'.format('.xlsx'))))):
        print('''######################## CRITICAL ERROR ####################\n
        The provided baseline file results for Graphics des not exist.....please see line 28 of configuration file\n
        section [GRAPHICS] [item] Baseline:''')
        exit()

    else:
        #Load the baseline data
        df_baseline = pd.read_excel(os.path.join(xls_outputs,''.join(('Results_',parser.get('GRAPHICS','Baseline'),'{}'.format('.xlsx')))))
        df_baseline['YEAR'] = pd.DatetimeIndex(df_baseline['DATE']).year
        df_baseline['DayofYear'] = pd.DatetimeIndex(df_baseline['DATE']).dayofyear

    #################################### PARAMETERS #########################################
    WL_StopCriterion=float(parser.get('COMPUTING','StopCriterion'))
    Initial_lake_waterLevel=float(parser.get('COMPUTING','Initial_lake_waterLevel'))
    Initial_river_waterLevel=float(parser.get('COMPUTING','Initial_river_waterLevel'))
    Initial_river_flow=float(parser.get('COMPUTING','Initial_river_flow'))

    HOURLY=False
    DAILY=False
    try:
        if parser.get('COMPUTING','time_steps')=='hourly':
            HOURLY=True
            Inital_Date=pd.to_datetime(parser.get('COMPUTING', 'Inital_Date'), format='%Y-%m-%d %H:%M:%S')
            col_Lake_level_hourly=parser.get('COMPUTING', 'Lake_level_hourly')
            col_Lake_level_hourly_QTM_AVG = parser.get('COMPUTING', 'Lake_level_hourly_QTM_AVG')

        else:
            if parser.get('COMPUTING','time_steps')=='daily':
                DAILY = True

            Inital_Date=pd.to_datetime(parser.get('COMPUTING','Inital_Date'), format='%Y-%m-%d', errors='ignore')
    except:
        print(r'''######################## CRITICAL ERROR ####################\n
               Error in the provided Inital date please. Make sure format is yyyy-mm-dd[space]HH:MM:SS .....please see line 28 of configuration file\n
               section [COMPUTING] [Inital_Date] equation_parameters:''')
        exit()

    ########################## HYDROLOGICAL SERIES #######################################################
    try:

        df_hydrological_serie = pd.read_excel(os.path.join(data_folder, parser.get('HYDROLOGICAL_DATA','serie')))
        df_hydrological_serie['Daily_datetime'] = pd.to_datetime(df_hydrological_serie['DATE'], format='yyyy-mm-dd', errors='ignore')
        df_hydrological_serie = df_hydrological_serie.set_index(['Daily_datetime'])

    except:
        print(r'''######################## CRITICAL ERROR ####################\n
                     The provided HYDROLOGICAL_DATA file does not exist.....please see line 9 of configuration file\n
        section [GRAPHICS] [item] Level_flow_Sorel_StOurs.xlsx:''')
        exit()




    ########################## HYDRAULIC GEMOETRY PARAMTERS #######################################################


    ########################### MANNING FILE LOADING ###########################
    #if DAILY:
    df_ST_RP_manning_daily = pd.read_excel(
        os.path.join(data_folder, 'hydraulics', parser.get('HYDRAULIC_GEOMETRY', 'ManningFile_daily')))
    ###### daily manning for 2011 ######################
    df_ST_RP_manning_daily_2011 = pd.read_excel(
        os.path.join(data_folder, 'hydraulics', parser.get('HYDRAULIC_GEOMETRY', 'ManningFile_daily_2011')))


#else:
    df_ST_RP_manning=pd.read_excel(os.path.join(data_folder,'hydraulics',parser.get('HYDRAULIC_GEOMETRY','ManningFile')))
    #print(df_ST_RP_manning.head())



    ########################### END MANNING FILE LOADING ###########################

    df_Lake_Volume = pd.read_excel(os.path.join(data_folder, parser.get('LAKE_STORAGE', 'file')))

    df_CrossSectionRousesPoint=pd.read_excel(os.path.join(data_folder,'hydraulics',parser.get('HYDRAULIC_GEOMETRY','CrossSectionFile')))
    #print(df_CrossSectionRousesPoint)

    distance_to_section=int(parser.get('HYDRAULIC_GEOMETRY','distance'))



    ######################################## NBS DATA ##########################################
    try:
        if NBS_type.lower()=='h':
            df_NBS = pd.read_excel(os.path.join(data_folder, 'NBS',parser.get('NBS_serie', 'file')))
        elif NBS_type.lower() == 'sto':
            df_NBS = pd.read_excel(os.path.join(data_folder, 'NBS','STOCHASTIC', parser.get('NBS_serie', 'file')))
        elif NBS_type.lower() == 'cc':

            df_NBS = pd.read_excel(os.path.join(data_folder, 'NBS','NBS_MODEL', parser.get('NBS_serie', 'file')))
            print('################################### WARNING CLIMATE CHANGE SCENARIO SIMULATION (takes longer) #################################\n')



        else:
            df_NBS=pd.read_excel(os.path.join(data_folder,'NBS',parser.get('NBS_serie','model'),'QTM_NBS',parser.get('NBS_serie','file')),header=None,names=['QTM','DATE','NBS'])
    except FileNotFoundError:
        print('''######################## CRITICAL ERROR ####################\n
        The provided Net basin Supply file file does not exist.....please see line 5 of configuration file\n
        section [NBS_serie] [file] file:''')
        exit()



### BOOLEAN ASSIGNEMENT
    STAGE_DISCHARGE=str_to_bool(parser.get('STAGE-DISCHARGE','equation'))
    #MARINA = str_to_bool(parser.get('MARINA', 'equation'))
    DIVERSION = False




    StageDischargeEquationParameters = []

    if not STAGE_DISCHARGE:
        #load the interpolator data
        df_Q_Interpolator=pd.read_excel(os.path.join(data_folder,'hydraulics',parser.get('STAGE-DISCHARGE','level_parameters_file')))

    else:

        StageDischargeEquationParameters=[]
        try:
            [StageDischargeEquationParameters.append(float(i)) for i in parser.get('STAGE-DISCHARGE','equation_parameters').split(',')]
            #Set The StageDischargeParamters
            _PARAMETERS=StageDischargeEquationParameters
        except ValueError:
            print('''######################## CRITICAL ERROR ####################\n
            Error in the provided equation parameters check for spaces, whiteline etc .....please see line 11 of configuration file\n
            section [STAGE-DISCHARGE] [item] equation_parameters:''')
            exit()


    marina_corrector_df = pd.read_excel(
        os.path.join(data_folder, 'hydraulics', parser.get('MARINA', 'marina_manning_file')))
    df_Marina_current = pd.read_excel(
        os.path.join(data_folder, 'hydraulics', parser.get('MARINA', 'level_parameters_file')))

    marina_LowFlow_slope_df=pd.read_excel(
        os.path.join(data_folder, 'hydraulics', parser.get('MARINA', 'marina_lowflow_estimator_file')))




    ###### CHECK ALTERNATIVE FOR WBM run
    ALTERNATIVE = parser.get('STAGE-DISCHARGE', 'alternative')
    if ALTERNATIVE not in ['CURRENT','CRUMP','CRUMP_DIV']:
        print('''######################## CRITICAL ERROR ####################\n
                   Error in the provided ALTERNATIVE CHECK THE AVAILABLE ALTERNATIVE , whiteline etc .....please see line 33 of configuration file\n
                   section [STAGE-DISCHARGE] [ALTERNATIVE] equation_parameters:''')
        exit()
    if ALTERNATIVE=='CRUMP_DIV':
        DIVERSION = True




    ###########################Output Dataframe ################################

    Out_df=df_NBS.copy()
    #Create the output column
    Out_df['LAKE_LEVEL']=-999.
    Out_df['RICH_FLOW_END']=-999.
    Out_df['RICH_LEVEL']=-999.


    print('######################### MODEL COMPUTING ######################################\n')

    ################################# OUTPUT FILE MANAGEMENT ###################################
    OPEN_IN_EXCEL=False
    try:
        OPEN_IN_EXCEL=str_to_bool(parser.get('RESULTS','OpenInExcel'))
        
    except:
        OPEN_IN_EXCEL=False

    ################################# GRAPHICS MANAGEMENT ###################################
    GENERATE_GRAPHS=False
    try:
        GENERATE_GRAPHS=str_to_bool(parser.get('GRAPHICS','Generate'))
    except:
        GENERATE_GRAPHS=False

    
    ################################# GRAPHICS MANAGEMENT ###################################
    GENERATE_DYNAMIC_GRAPHS=False
    try:
        GENERATE_DYNAMIC_GRAPHS=str_to_bool(parser.get('GRAPHICS','GenerateDynamic'))
    except:
        GENERATE_DYNAMIC_GRAPHS=False

    ##########################################################################



    #check for DOWNSTREAM FLOWS AND LEVELS
    DOWNSTREAM = False
    if parser.get('DOWNSTREAM','level')!= 'None':
        DOWNSTREAM=True


    ####### progress bar ##############
    CONTROL='CREST'


    with tqdm(total=len(list(df_NBS.iterrows()))) as pbar:
        for index,row in df_NBS.iterrows():



            pbar.update(1)

            #Set the inital condition at the start of WBM (Model Initialization) 
            if index==0:
                _Lake_volume_start_qtm= LakeVolumeInterpolator(Initial_lake_waterLevel,df_Lake_Volume['LakeLevel'],df_Lake_Volume['LakeVolume'])
                _River_flow=Initial_river_flow
                Previous_qtm_date=Inital_Date
                Final_iteration_river_level=Initial_river_waterLevel
                Final_iteration_lake_level=Initial_lake_waterLevel
                _River_flow_start_qtm=Initial_river_flow
                
                _Lake_Level_start_qtm=Initial_lake_waterLevel
                _River_Level_start_qtm=Initial_river_waterLevel
            else:
                #Set the inital condition for the QTM

                if HOURLY:
                    Previous_qtm_date = df_NBS.iloc[index - 1]['Hourly_datetime']
                else:
                    Previous_qtm_date=df_NBS.iloc[index-1]['DATE']


                _Lake_volume_start_qtm= LakeVolumeInterpolator(_Lake_Level_start_qtm,df_Lake_Volume['LakeLevel'],df_Lake_Volume['LakeVolume'])
                
###################################### MANNING SELECTION ###########################################################
            #Get the Manning for the current QTM
            #for QTM runs
            day_of_year = int(row.DATE.timetuple().tm_yday)
            if DAILY:
                day_of_year = int(row.DATE.timetuple().tm_yday)
                year=int(row.DATE.timetuple().tm_year)



                ###### IF WE ARE TESTING DIVERSION AND YEAR ==2011,we are using the manning of 2011#######
                special_year=-999
                if parser.get('COMPUTING', 'special_year') !='None':
                    special_year=parser.get('COMPUTING', 'special_year')



                if special_year==str(2011):

                    #testing baseline for 2011
                    _Manning = float(df_ST_RP_manning_daily_2011.at[(day_of_year - 1), 'manning'])
                    WBM_logger.info(
                        '########################## Manning is daily 2011 #####################################')


                else:


                    _Manning = float(df_ST_RP_manning.at[int((row['QTM'] - 1)), 'manning'])

                    WBM_logger.info(
                    '########################## Manning is daily avg 2010-2016 #####################################')

            else:

                _Manning=float(df_ST_RP_manning.at[int((row['QTM']-1)),'manning'])
                WBM_logger.info(
                    '########################## Manning is QTM avg 2010-2016 #####################################')




###################################### END MANNING SELECTION ###########################################################

            #Get the Net Basin Supply for the current QTM
            _NBS=row['NBS']

            #TimeDifference since last QTM (some QT have 7 and other 8 days)
            if HOURLY:
                _diff_time_in_seconds = row['Hourly_datetime'] - Previous_qtm_date
            else:
                _diff_time_in_seconds=row['DATE']-Previous_qtm_date

            _diff_time_in_seconds=_diff_time_in_seconds.total_seconds()


            #Inital_guess_diff
            LakeLevelDiff=1.
            SaintJeanDiff=1.
            #Iterate the lake volume and level
            loop_lake=0
            loop=0

            if HOURLY:
                WBM_logger.info(
                    '!!!!!!!!!!!!!!!!!!!!!!!!!!! current DATE : {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(row.Hourly_datetime))
            else:

                WBM_logger.info('!!!!!!!!!!!!!!!!!!!!!!!!!!! current DATE : {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(row.DATE))


            _PARAMETERS = StageDischargeEquationParameters
            #_PARAMETERS_MARINA=MarinaEquationParameters



            WBM_logger.info('STARGE DISCHARGE PARAMETERS FOR THIS QTM : {}'.format(_PARAMETERS))

        ##########################################################################################################################################################

            while LakeLevelDiff >= WL_StopCriterion:
                if loop_lake==0:
                    previousRousesPointLevel=Final_iteration_lake_level
                    _Lake_volume_iteration=(-(_River_flow_start_qtm+_River_flow_start_qtm)/2+_NBS)*_diff_time_in_seconds+_Lake_volume_start_qtm
                    _Lake_Level_iteration=LakeLevleInterpolatorFromVolume(_Lake_volume_iteration,df_Lake_Volume['LakeVolume'],df_Lake_Volume['LakeLevel'])
                
                else:
                    previousRousesPointLevel=_Lake_Level_iteration

        ################# LAKE LEVEL FROM LAKE VOLUME ###########################
                    _Lake_volume_iteration=(-(_River_flow_iteration+_River_flow_start_qtm)/2+_NBS)*_diff_time_in_seconds+_Lake_volume_start_qtm
                    #print(_Lake_volume_iteration)
                    _Lake_Level_iteration=LakeLevleInterpolatorFromVolume(_Lake_volume_iteration,df_Lake_Volume['LakeVolume'],df_Lake_Volume['LakeLevel'])
                    #print(_Lake_Level_iteration)


                #Get all the hydraulic geometry parameter of the section
                _CrossSectionArea=CrossSectionAreaInterpolator(_Lake_Level_iteration,df_CrossSectionRousesPoint['LEVEL'],df_CrossSectionRousesPoint['AREA'])
                _WettedPerimeter=WettedPerimeterInterpolator(_Lake_Level_iteration,df_CrossSectionRousesPoint['LEVEL'],df_CrossSectionRousesPoint['PERIMETER'])
                #Iterate over River flows and levels

                
                while SaintJeanDiff >= WL_StopCriterion:
                    if loop==0:
                        previousSaintJeanLevel=Final_iteration_river_level

                        #Find the water level of the river from the lake level and river flows, and stage-discharge relationship
                        #LEVEL
                        _Level_ST_JEAN_iteration=Get_WaterLevel_SaintJean_geom(_River_Level_start_qtm,distance_to_section,_Lake_Level_iteration,_River_flow_start_qtm,_Manning,_CrossSectionArea,_WettedPerimeter)
                        #FLOW

                        if not STAGE_DISCHARGE:


################################# condition to test with or without diversion #####################################

                                if DIVERSION:  ## THE ALTERNATIVE IS BY DEFAULT CRUMP_DIV
                                    if _River_flow_start_qtm >=1012:  # DIversion is open
                                        CONTROL='SHOAL/CREST DIVERSION'
                                        _River_flow_iteration = (FlowInterpolator(_Level_ST_JEAN_iteration,
                                                                                  df_Q_Interpolator[
                                                                                      'H_SJ_VIRT_NS_CRUMP_DIV2D'],
                                                                                  df_Q_Interpolator['FLOW_NS_CRUMP_DIV2D'], WBM_logger,
                                                                                  row['DATE']) + _River_flow_start_qtm) / 2
                                    else:
                                        CONTROL='SHOAL/CREST'
                                        _River_flow_iteration = (FlowInterpolator(_Level_ST_JEAN_iteration,
                                                                                  df_Q_Interpolator[
                                                                                      'H_SJ_VIRT_DELTA_COR_NS'],
                                                                                  df_Q_Interpolator['FLOW_NS'], WBM_logger,



                                                                                  row['DATE']) + _River_flow_start_qtm) / 2
                                else: #ALTERNATIVE COULD BE O or CRUMP

                                    if ALTERNATIVE=='CURRENT':

                                        CONTROL='SHOAL'

                                        _River_flow_iteration = (FlowInterpolator(_Level_ST_JEAN_iteration,
                                                                                  df_Q_Interpolator[
                                                                                      'H_SJ_VIRT_O'],
                                                                                  df_Q_Interpolator['FLOW'], WBM_logger,
                                                                                  row['DATE']) + _River_flow_start_qtm) / 2


                                    else:    # ALTERNATIVE IS CRUMP

                                        CONTROL = 'SHOAL/CREST'
                                        _River_flow_iteration = (FlowInterpolator(_Level_ST_JEAN_iteration,
                                                                                  df_Q_Interpolator[
                                                                                      'H_SJ_VIRT_DELTA_COR_NS'],
                                                                                  df_Q_Interpolator['FLOW_NS'], WBM_logger,
                                                                                  row['DATE']) + _River_flow_start_qtm) / 2




################################# end condition to test with or without diversion #####################################

                    else:
                        previousSaintJeanLevel=_Level_ST_JEAN_iteration

                        
                        #Find the water level of the river from the lake level and river flows, and stage-discharge relationship

                        _Level_ST_JEAN_iteration=Get_WaterLevel_SaintJean_geom(previousSaintJeanLevel,distance_to_section,_Lake_Level_iteration,_River_flow_iteration,_Manning,_CrossSectionArea,_WettedPerimeter)                
                        #FLOW



                        if not STAGE_DISCHARGE:


################################# condition to test with or without diversion #####################################
                                if DIVERSION: ## THE ALTERNATIVE IS BY DEFAULT CRUMP_DIV
                                    if _River_flow_iteration >=1012: # DIversion is open
                                        CONTROL='SHOAL/CREST DIVERSION'  #SHIFTED EQUATION (1200 cms and up)
                                        _River_flow_iteration = (FlowInterpolator(_Level_ST_JEAN_iteration,
                                                                          df_Q_Interpolator['H_SJ_VIRT_NS_CRUMP_DIV2D'],
                                                                          df_Q_Interpolator['FLOW_NS_CRUMP_DIV2D'], WBM_logger,
                                                                          row['DATE']) + _River_flow_iteration) / 2

                                    else:
                                        CONTROL = 'SHOAL/CREST'  #
                                        _River_flow_iteration = (FlowInterpolator(_Level_ST_JEAN_iteration,
                                                                                  df_Q_Interpolator['H_SJ_VIRT_DELTA_COR_NS'],
                                                                                  df_Q_Interpolator['FLOW_NS'], WBM_logger,
                                                                                  row['DATE']) + _River_flow_iteration) / 2

                                else: #ALTERNATIVE COULD BE O or CRUMP

                                    if ALTERNATIVE == 'CURRENT':

                                        CONTROL='SHOAL'

                                        _River_flow_iteration = (FlowInterpolator(_Level_ST_JEAN_iteration,
                                                                                  df_Q_Interpolator[
                                                                                      'H_SJ_VIRT_O'],
                                                                                  df_Q_Interpolator['FLOW'], WBM_logger,
                                                                                  row['DATE']) + _River_flow_start_qtm) / 2

                                    else:

                                        CONTROL = 'SHOAL/CREST'
                                        _River_flow_iteration = (FlowInterpolator(_Level_ST_JEAN_iteration,
                                                                                  df_Q_Interpolator['H_SJ_VIRT_DELTA_COR_NS'],
                                                                                  df_Q_Interpolator['FLOW_NS'], WBM_logger,
                                                                                  row['DATE']) + _River_flow_iteration) / 2




                    #Check the WL_StopCriterion
                    
                    SaintJeanDiff=abs(previousSaintJeanLevel-_Level_ST_JEAN_iteration)

                    #reasign value for  the next iteration
                    loop+=1
                    if loop>100:
                        SaintJeanDiff=WL_StopCriterion-1.0


                
                #Check the WL_StopCriterion for the Lake Level
                
                LakeLevelDiff=abs(previousRousesPointLevel-_Lake_Level_iteration)


                ##reasign value for  the next Lake iteration
                loop_lake+=1
                if loop_lake>10:
                    LakeLevelDiff=WL_StopCriterion-1.0    


            #Set the final values for next qtm

            _Lake_Level_start_qtm=_Lake_Level_iteration

            #2_Inital_River_flow


            _River_flow_start_qtm=_River_flow_iteration

            #3_Inital_Lake_Volume
            _Lake_volume_start_qtm=_Lake_volume_iteration
            #print(_Lake_volume_start_qtm)
            #4_Inital_River_Level_start_qtm
            _River_Level_start_qtm=_Level_ST_JEAN_iteration

            Final_iteration_lake_level=_Lake_Level_iteration
            Final_iteration_river_level=_Level_ST_JEAN_iteration
            Final_iteration_river_flow = _River_flow_iteration


            #print(row['DATE'],_Lake_Level_iteration)

            WBM_logger.info('!!!!!!!!!!!!!!!!!!!!!!!!!!! FINAL QTM LAKE LEVEL : {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'.format(_Lake_Level_iteration))
            Out_df.at[index,'LAKE_LEVEL']=_Lake_Level_iteration
            Out_df.at[index,'RICH_FLOW_END']=_River_flow_iteration
            Out_df.at[index,'RICH_LEVEL']=_Level_ST_JEAN_iteration

            #Print the Water level at Sorel for that day


            ####bLOCK OF DATA FFO DOWNSTREAM LEVEL AND FLOWS #############################

            if DOWNSTREAM:
                hydro_daily_values=df_hydrological_serie[df_hydrological_serie['DATE'] == row.DATE]


                H_SOREL=hydro_daily_values.H_SOREL.values[0]
                SAINT_OURS=(hydro_daily_values.TRIB.values[0])+_River_flow_iteration
                Q_FRYER_SO=hydro_daily_values.TRIB.values[0]

                if Q_FRYER_SO<0:
                    Q_FRYER_SO=0

                Out_df.at[index, 'H_SOREL'] = H_SOREL

                #####FIND the difference of discharge of the river compared to Baseline

                #flow_diff = _River_flow_iteration- df_baseline.iloc[index].RICH_FLOW_END
                Out_df.at[index, 'Q_FRYER_SO'] = Q_FRYER_SO

                Out_df.at[index, 'Q_ST_OURS'] =SAINT_OURS




            #Interpolate the value at Saint-Jean Marina




            if CONTROL == 'SHOAL':

                _marina_level = MarinaLevelInterpolator(_Level_ST_JEAN_iteration, df_Marina_current['H_SJ_VIRT_O'],
                                                        df_Marina_current['H_SJ_MARINA_O'],WBM_logger,row['DATE'])

            if CONTROL == 'SHOAL/CREST':

                _marina_level = MarinaLevelInterpolator(_Level_ST_JEAN_iteration, df_Marina_current['H_SJ_VIRT_DELTA_COR_NS'],
                                                        df_Marina_current['H_SJ_MARINA_NS'],WBM_logger,row['DATE'])


            if CONTROL == 'SHOAL/CREST DIVERSION':



                _marina_level = MarinaLevelInterpolator(_Level_ST_JEAN_iteration, df_Marina_current['H_SJ_VIRT_NS_CRUMP_DIV2D'],
                                                        df_Marina_current['H_SJ_MARINA_NS_CRUMP_DIV2D'],WBM_logger,row['DATE'])



            #_marinalevel_before_correction=_marina_level
            _marina_level=_marina_level+(marina_corrector_df.at[(day_of_year - 1), 'diff'])


            Out_df.at[index, 'LEVEL_MARINA'] = _marina_level
            Out_df.at[index, 'CONTROL'] = CONTROL
            Out_df.at[index, 'Q_CANAL'] =np.nan


    
        ####### 2022-02-23 ###### correction for low flow water levele estimation at Saint-Jean Marina
        Out_df['LEVEL_MARINA'] = Out_df.apply(MarinaLevelCorrector, args=(marina_LowFlow_slope_df, ALTERNATIVE,), axis=1)




        print('\nGENERATING OUTPUTS')
        try:
            Out_df.to_excel(os.path.join(xls_outputs,''.join('Results_{}_{}.xlsx'.format(NBS_supply_scenario,Shoal_modification))))
            print('\nGENERATING OUTPUTS')
        except PermissionError:
            print('''######################## CRITICAL ERROR ####################\n
            This file:{} is open in Excel. Please close it and rerun the model \n'''.format(os.path.join(xls_outputs,''.join('Results_{}_{}.xlsx'.format(NBS_supply_scenario,Shoal_modification)))))
            exit()

        #Open the Excel file again
        if OPEN_IN_EXCEL:
            xls_file=os.path.join(xls_outputs,''.join('Results_{}_{}.xlsx'.format(NBS_supply_scenario,Shoal_modification)))
            os.system('start excel.exe {}'.format(xls_file))


        print('\n')
        if GENERATE_GRAPHS:
            print('GENERATING GRAPHICS')
            ########## graph test##############
            grapher=WBMGrapher.WBM_PLOTTER(parser=parser)

        if GENERATE_DYNAMIC_GRAPHS:
            print('GENERATING HTML GRAPHICS')
            ########## graph test##############
            if NBS_type.lower() == 'cc':
                DynamicGrapher=WBMDynamicPlotter_CC.WBM_DYNAMIC_PLOTTER(parser=parser)
            else:
                DynamicGrapher=WBMDynamicPlotter.WBM_DYNAMIC_PLOTTER(parser=parser,sector='LAKE')
                del DynamicGrapher
                DynamicGrapher = WBMDynamicPlotter.WBM_DYNAMIC_PLOTTER(parser=parser, sector='RIVER')
                del DynamicGrapher
                DynamicFlowGrapher = WBMDynamicFlowPlotter.WBM_DYNAMIC_FLOW_PLOTTER(parser=parser, sector='RIVER')
                del DynamicFlowGrapher


    #copy the log file into output folder
    if USER_FOLDER:
        #copy the verison of the WBM file used to generate the results
        print('__file__:    ', __file__)
        shutil.copy(__file__,os.path.join(user_output_folder,__file__.split(os.sep)[-1]))

        shutil.copy(os.path.join(xls_outputs,''.join('Results_{}_{}.xlsx'.format(NBS_supply_scenario,Shoal_modification))),os.path.join(user_output_folder,''.join('Results_{}_{}.xlsx'.format(NBS_supply_scenario,Shoal_modification))))

        if GENERATE_DYNAMIC_GRAPHS:
            shutil.copy(os.path.join(graphs_folder,''.join(('Results_{}_{}_{}.html'.format(NBS_supply_scenario, Shoal_modification,'LAKE_CHAMPLAIN')))),
                    os.path.join(user_output_folder,''.join(('Results_{}_{}_{}.html'.format(NBS_supply_scenario, Shoal_modification, 'LAKE_CHAMPLAIN')))))

            shutil.copy(os.path.join(graphs_folder, ''.join(
                ('Results_{}_{}_{}.html'.format(NBS_supply_scenario, Shoal_modification, 'RICHELIEU_RIVER')))),
                         os.path.join(user_output_folder, ''.join(
                    ('Results_{}_{}_{}.html'.format(NBS_supply_scenario, Shoal_modification, 'RICHELIEU_RIVER')))))
            shutil.copy(os.path.join(graphs_folder, ''.join(
                ('Results_{}_{}_{}.html'.format(NBS_supply_scenario, Shoal_modification, 'RICHELIEU_RIVER_flow')))),
                         os.path.join(user_output_folder, ''.join(
                    ('Results_{}_{}_{}.html'.format(NBS_supply_scenario, Shoal_modification, 'RICHELIEU_RIVER_flow')))))

        shutil.copy(os.path.join(log_folder, ''.join('WBM_{}.log'.format(Shoal_modification))),
                    os.path.join(user_output_folder, ''.join('WBM_{}.log'.format(Shoal_modification))))



    print(f'LAKE CHAMPLAIN WATER BALANCE MODEL version {__version__} COMPUTING DONE')


if __name__ == "__main__":
    main(sys.argv[1:])

    


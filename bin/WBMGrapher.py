# -*- coding: UTF-8 -*-
####################################################################
__author__ = 'Section Hydrologie et Ecohydraulique--Olivier Champoux@canada.ca'
__version__ = '1.0.0'
__date__ = '2019-09-30'
__update__ = ''
__copyright__ = 'Copyright (c) 2019 Environment Canada'
# *****************************************************************************
import os,sys
import pandas as pd
import matplotlib.pyplot as plt
import configparser
import subprocess

class WBM_PLOTTER(object):
    _WBM_PLOTTER_VERSION='1.0.0'
    _parser=None
    _graphs_folder=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'output','graphs')))
    _xls_outputs=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'output','Excel_files')))
    _xls_outputs_base=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'output','Excel_files')))
    _base_file_results=None
    _comparaison_results=None

    def str_to_bool(self,s):
        if s == 'True':
            return True
        elif s == 'False':
            return False
        else:
            raise ValueError # evil ValueError that doesn't tell you what the wrong value was


    def __init__(self,parser=None):
        self._parser=parser
        

        ###### get screen resolution #############
        cmd='wmic path Win32_VideoController get CurrentHorizontalResolution,CurrentVerticalResolution'
        p1 = subprocess.Popen(cmd, shell=True, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p1.communicate()
        CurrentHorizontalResolution=int(out.split()[-2])
        CurrentVerticalResolution=int(out.split()[-1])
        my_dpi=int(self._parser.get('GRAPHICS','DPI'))
        plt.figure(figsize=(CurrentHorizontalResolution/my_dpi, CurrentVerticalResolution/my_dpi), dpi=my_dpi)


        
        ###### read the parser #########
        NBS_supply_scenario=self._parser.get('NBS_serie','name')
        Shoal_modification=self._parser.get('STAGE-DISCHARGE','name')

        self._base_file_results=os.path.join(self._xls_outputs_base,''.join(('Results_',self._parser.get('GRAPHICS','Baseline'),'{}'.format('.xlsx'))))
        self._comparaison_results=os.path.join(self._xls_outputs,''.join(('Results_{}_{}.xlsx'.format(NBS_supply_scenario,Shoal_modification))))


        ############################################## results ######################################
        BASE_SITUATION_df=pd.read_excel(self._base_file_results)
        Out_df=pd.read_excel(self._comparaison_results)



        ###### Verification##########
        #ax2 = Out_df.plot.scatter(x='LAKE_LEVEL_OBS',y='Current',color='blue',marker=',',label='python')
        #Old_results.plot.scatter(x='LAKE_LEVEL_OBS',y='LAKE_LEVEL',color='red',marker='.',label='excel',ax=ax2)
        #merge_df=pd.merge(Out_df, Old_results, on='DATE')
        #print merge_df.head()
        #ax3=merge_df.plot.scatter(x='LAKE_LEVEL',y='Current',color='red',marker='.')
        #plt.show()
        #quit()

        #ax2.legend()
        #plt.show()
        #quit()

        ###### date index ##############
        BASE_SITUATION_df['DATE'] =  pd.to_datetime(BASE_SITUATION_df['DATE'])
        BASE_SITUATION_df.set_index('DATE', inplace=True)

        Out_df['DATE'] =  pd.to_datetime(Out_df['DATE'])
        Out_df.set_index('DATE', inplace=True)


        

        #Out_df['LAKE_LEVEL_OBS'].plot(ax=ax1,label='LAKE_LEVEL_OBS',color='black',linestyle="-",linewidth=2)
        ######################################COMP SITUATION PLOTTING##############################

        ax1=Out_df['LAKE_LEVEL'].plot(label=self._parser.get('GRAPHICS','Calculated_legend_label'),
        color=self._parser.get('GRAPHICS','Calculated_line_color'),
        linestyle="-",
        linewidth=int(self._parser.get('GRAPHICS','Calculated_line_width')))
      

        #######################################BASE SITUATION PLOTTING##############################
        BASE_SITUATION_df['LAKE_LEVEL'].plot(ax=ax1,
        label=self._parser.get('GRAPHICS','Baseline_legend_label'),
        color=self._parser.get('GRAPHICS','Baseline_line_color'),
        linestyle="--",
        linewidth=int(self._parser.get('GRAPHICS','Baseline_line_width')))



        ####################################### LABELS ###########################################
        title=self._parser.get('GRAPHICS','Graph_title')
        ylabel='Lake level (m-NAVD88)'
        xlabel='Date'
        ax1.set_ylabel(ylabel,fontname='consolas', fontsize=12)
        ax1.set_xlabel(xlabel,fontname='consolas', fontsize=12)
        ax1.set_title(title, fontname='consolas', fontsize=12)
        ttl = ax1.title
        ttl.set_position([.5, 1.05])
        ax1.legend()

        GENERATE_GRAPHS=self.str_to_bool(parser.get('GRAPHICS','Generate'))
        SHOW_GRAPHS=self.str_to_bool(parser.get('GRAPHICS','Show'))

        if GENERATE_GRAPHS:
            graph_file=os.path.join(self._graphs_folder,''.join(('Results_{}_{}.png'.format(NBS_supply_scenario,Shoal_modification))))
            plt.savefig(graph_file)
            
  
        if SHOW_GRAPHS:
            plt.show()
  
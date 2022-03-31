# -*- coding: UTF-8 -*-
####################################################################
__author__ = 'Section Hydrologie et Ecohydraulique--Olivier Champoux@canada.ca'
__version__ = '1.0.0'
__date__ = '2019-09-30'
__update__ = ''
__copyright__ = 'Copyright (c) 2019 Environment Canada'
# *****************************************************************************
import os
import numpy as np
from bokeh.io import export_png
from bokeh.io import show, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool,HoverTool
from bokeh.plotting import figure
import pandas as pd
import configparser
import subprocess

class WBM_DYNAMIC_PLOTTER(object):
    WBM_DYNAMIC_PLOTTER='1.0.0'
    _parser=None
    _graphs_folder=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'output','html_graphs',)))
    _xls_outputs=(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'output','Excel_files')))
    _base_file_results=None
    _comparaison_results=None

    


    def __init__(self,parser=None):
        self._parser=parser
        



        ###### get screen resolution #############
        cmd='wmic path Win32_VideoController get CurrentHorizontalResolution,CurrentVerticalResolution'
        p1 = subprocess.Popen(cmd, shell=True, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p1.communicate()
        CurrentHorizontalResolution=int(out.split()[-2])
        CurrentVerticalResolution=int(out.split()[-1])
        my_dpi=int(self._parser.get('GRAPHICS','DPI'))
        #plt.figure(figsize=(CurrentHorizontalResolution/my_dpi, CurrentVerticalResolution/my_dpi), dpi=my_dpi)


        
        ###### read the parser #########
        NBS_supply_scenario=self._parser.get('NBS_serie','name')
        Shoal_modification=self._parser.get('STAGE-DISCHARGE','name')
        

        output_file(os.path.join(self._graphs_folder,''.join(('Results_{}_{}.html'.format(NBS_supply_scenario,Shoal_modification)))))

        self._base_file_results=os.path.join(self._xls_outputs,''.join(('Results_',self._parser.get('GRAPHICS','Baseline'),'{}'.format('.xlsx'))))
        self._comparaison_results=os.path.join(self._xls_outputs,''.join(('Results_{}_{}.xlsx'.format(NBS_supply_scenario,Shoal_modification))))


        ############################################## results ######################################
        BASE_SITUATION_df=pd.read_excel(self._base_file_results)######LAKE_LEVEY_y in MERGE_DF
        Out_df=pd.read_excel(self._comparaison_results)######LAKE_LEVEY_x in MERGE_DF

        dates_Out_df = np.array(Out_df['DATE'], dtype=np.datetime64)
        dates_baseline = np.array(BASE_SITUATION_df['DATE'], dtype=np.datetime64)
        Out_df["DateString"] = Out_df["DATE"].dt.strftime("%Y-%m-%d")
        BASE_SITUATION_df["DateString"] = BASE_SITUATION_df["DATE"].dt.strftime("%Y-%m-%d")



        
        #merge dataframe
        #merge_df=pd.merge(Out_df,BASE_SITUATION_df,how='inner', on='DATE')
        #print(merge_df.head())

        # create a string column for plotting
        #merge_df["DateString"] = merge_df["DATE"].dt.strftime("%Y-%m-%d")
        #merge_df["Diff"] = merge_df["LAKE_LEVEL_x"]-merge_df["LAKE_LEVEL_y"]


        
        HoverTool(tooltips=[('DATE', '@DateTime{%F}')],
    formatters={'DateTime': 'datetime'})

        hover = HoverTool(
            tooltips=[
                ("DATE", '@DateTime{%F}'),
                (self._parser.get('GRAPHICS','Calculated_legend_label'), "@LAKE_LEVEL")
            ],formatters={'DateTime': 'datetime'}
        )
    




        source_calc = ColumnDataSource(Out_df)
        source_base = ColumnDataSource(BASE_SITUATION_df)
   
  
     
        p = figure(plot_height=int(CurrentVerticalResolution*0.55), plot_width=int(CurrentHorizontalResolution*0.95), tools=[hover,"xpan","wheel_zoom","reset","box_zoom"], toolbar_location="above",
                x_axis_type="datetime", x_axis_location="above",
                background_fill_color="#efefef", x_range=(dates_Out_df[1500], dates_Out_df[2500]))

        red_renderer_line1 =p.line('DATE', 'LAKE_LEVEL', source=source_calc,color='#A6CEE3',line_width=1, legend=self._parser.get('GRAPHICS','Calculated_legend_label'))
        red_renderer_line2 =p.line('DATE', 'LAKE_LEVEL', source=source_base, color='#FB9A99',line_width=1, legend=self._parser.get('GRAPHICS','Baseline_legend_label'))
        
        
        p.add_tools(HoverTool(tooltips=[("DATE", '@DateString'),
        (self._parser.get('GRAPHICS','Calculated_legend_label'), "@LAKE_LEVEL")],
        formatters={'date':'datetime'},mode="vline",renderers=[red_renderer_line1]))

        p.add_tools(HoverTool(tooltips=[("DATE", '@DateString'),
        (self._parser.get('GRAPHICS','Baseline_legend_label'), "@LAKE_LEVEL")],
        formatters={'date':'datetime'},mode="vline",renderers=[red_renderer_line2]))


        p.yaxis.axis_label = 'Lake Champlain water level (m-NAVD88)'
        p.grid.grid_line_alpha = 0
        p.ygrid.band_fill_color = "olive"
        p.ygrid.band_fill_alpha = 0.1

        select = figure(title="Drag the middle and edges of the selection box to change the range above",
                        plot_height=int(CurrentVerticalResolution*0.3), plot_width=int(CurrentHorizontalResolution*0.95), y_range=p.y_range,
                        x_axis_type="datetime", y_axis_type=None,
                        tools="", toolbar_location=None, background_fill_color="#efefef")

        range_tool = RangeTool(x_range=p.x_range)
        range_tool.overlay.fill_color = "navy"
        range_tool.overlay.fill_alpha = 0.2

        select.line('DATE', 'LAKE_LEVEL', source=source_calc)
        select.ygrid.grid_line_color = None
        select.add_tools(range_tool)
        select.toolbar.active_multi = range_tool

        show(column(p, select))
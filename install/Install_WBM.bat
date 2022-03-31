rem echo off
rem cls
set curDir=%CD%
cd /D %curDir%

call %UserProfile%\Miniconda3\Scripts\activate
call %UserProfile%\Miniconda3\Scripts\conda env create -f LCRR_WBM.yml
	

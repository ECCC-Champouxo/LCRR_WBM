echo off
cls
set curDir=%CD%
cd %curDir%WBM

setlocal enableDelayedExpansion


IF EXIST %LocalAppData%\Continuum\miniconda3\ (
	call %LocalAppData%\Continuum\miniconda3\Scripts\activate LCRR_WBM
	
	
) ELSE (
	call C:\Users\%USERNAME%\Miniconda3\Scripts\activate LCRR_WBM
	
	
)

python 
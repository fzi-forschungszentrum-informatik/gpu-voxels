@echo off

:GETTEMPNAME
set TMPFILE=%TMP%\setenv-%RANDOM%-%TIME:~6,2%%TIME:~9,2%.bat
if exist "%TMPFILE%" GOTO :GETTEMPNAME 

python script\ic\setenv.py --shell=cmd %* > %TMPFILE%
call %TMPFILE%
del %TMPFILE%

@echo off
setlocal enabledelayedexpansion

:: Set the SSH username
set ssh_user=ebrooks

:: Set the scheduler host
set scheduler_host=odd01.cs.ohio.edu
set scheduler_port=8786

:: Start the scheduler on the first machine in a new command prompt
echo Starting scheduler on %scheduler_host%...
start cmd /k "ssh %ssh_user%@%scheduler_host% python3 -m distributed.cli.dask_scheduler --host %scheduler_host%"

:: Wait for a few seconds to ensure scheduler starts
timeout /t 10 /nobreak

:: Start the workers on all machines listed in workers.txt in new command prompts
for /f "tokens=1" %%i in (.\distributed_setup\workers.txt) do (
    if "%%i" neq "%scheduler_host%" (
        echo Starting worker on %%i...
        start cmd /k "ssh %ssh_user%@%%i python3 -m distributed.cli.dask_worker tcp://%scheduler_host%:%scheduler_port%"
    )
)

:: End the script
echo Cluster setup complete. Check the logs for more details.
endlocal

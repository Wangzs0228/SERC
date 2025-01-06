cd E:\zts\HSI_domain_adaptation

for /l %%i in (1,1,10) do (
    call "train\dan\houston.bat" %%i
    call "train\dann\houston.bat" %%i
    @REM call "train\ddc\houston.bat" %%i
    @REM call "train\deepcoral\houston.bat" %%i
    call "train\dsan\houston.bat" %%i
    @REM call "train\jan\houston.bat" %%i
    call "train\mcd\houston.bat" %%i
    @REM call "train\nommd\houston.bat" %%i
    @REM call "train\self_training\houston.bat" %%i
)
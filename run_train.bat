@echo off
REM ============================================================
REM  Sequential RT-DETR Universal Cloak training
REM  overwatch -> cf -> cs2
REM  Each run BLOCKS until finished before the next one starts
REM  (this is the default behavior of a .bat file run top to
REM  bottom — no "start" command is used anywhere, so there is
REM  no parallel/overlapping GPU usage between runs).
REM ============================================================
setlocal

set MODEL=rtdetr
set EPOCHS=100
set EPSILON=8
set LR=0.0003

echo ============================================================
echo  [1/3] Training: overwatch / %MODEL%
echo  epochs=%EPOCHS%  epsilon=%EPSILON%/255  lr=%LR%
echo ============================================================
python train_cloak.py --game overwatch --model %MODEL% --n_iter %EPOCHS% --lr %LR% --epsilon %EPSILON% --data_path data\overwatch\frames
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed for overwatch. Stopping here so cf/cs2 don't run on a broken state.
    pause
    exit /b 1
)
echo.
echo  [1/3] Done: overwatch
echo.

echo ============================================================
echo  [2/3] Training: cf / %MODEL%
echo  epochs=%EPOCHS%  epsilon=%EPSILON%/255  lr=%LR%
echo ============================================================
python train_cloak.py --game cf --model %MODEL% --n_iter %EPOCHS% --lr %LR% --epsilon %EPSILON% --data_path data\cf\frames
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed for cf. Stopping here so cs2 doesn't run on a broken state.
    pause
    exit /b 1
)
echo.
echo  [2/3] Done: cf
echo.

echo ============================================================
echo  [3/3] Training: cs2 / %MODEL%
echo  epochs=%EPOCHS%  epsilon=%EPSILON%/255  lr=%LR%
echo ============================================================
python train_cloak.py --game cs2 --model %MODEL% --n_iter %EPOCHS% --lr %LR% --epsilon %EPSILON% --data_path data\cs2\frames
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed for cs2.
    pause
    exit /b 1
)
echo.
echo  [3/3] Done: cs2
echo.

echo ============================================================
echo  All 3 training runs complete!
echo    universal_cloak\overwatch\%MODEL%\universal_noise.pt
echo    universal_cloak\cf\%MODEL%\universal_noise.pt
echo    universal_cloak\cs2\%MODEL%\universal_noise.pt
echo ============================================================
pause
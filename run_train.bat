@echo off
REM ============================================================
REM  Sequential NanoDet Universal Cloak re-training
REM  (using the fixed max-confidence loss_nanodet)
REM  cf -> overwatch -> valorant
REM  Each run BLOCKS until finished before the next one starts
REM  (default behavior of a .bat file run top to bottom — no
REM  "start" command is used, so there is no parallel/overlapping
REM  GPU usage between runs).
REM ============================================================
setlocal

set MODEL=nanodet
set EPOCHS=100
set EPSILON=8
set LR=0.0005

echo ============================================================
echo  [1/3] Training: cf / %MODEL%
echo  epochs=%EPOCHS%  epsilon=%EPSILON%/255  lr=%LR%
echo ============================================================
python train_cloak.py --game cf --model %MODEL% --n_iter %EPOCHS% --lr %LR% --epsilon %EPSILON% --data_path data\cf\frames
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed for cf. Stopping here so overwatch/valorant don't run on a broken state.
    pause
    exit /b 1
)
echo.
echo  [1/3] Done: cf
echo.

echo ============================================================
echo  [2/3] Training: overwatch / %MODEL%
echo  epochs=%EPOCHS%  epsilon=%EPSILON%/255  lr=%LR%
echo ============================================================
python train_cloak.py --game overwatch --model %MODEL% --n_iter %EPOCHS% --lr %LR% --epsilon %EPSILON% --data_path data\overwatch\frames
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed for overwatch. Stopping here so valorant doesn't run on a broken state.
    pause
    exit /b 1
)
echo.
echo  [2/3] Done: overwatch
echo.

echo ============================================================
echo  [3/3] Training: valorant / %MODEL%
echo  epochs=%EPOCHS%  epsilon=%EPSILON%/255  lr=%LR%
echo ============================================================
python train_cloak.py --game valorant --model %MODEL% --n_iter %EPOCHS% --lr %LR% --epsilon %EPSILON% --data_path data\valorant\frames
if errorlevel 1 (
    echo.
    echo [ERROR] Training failed for valorant.
    pause
    exit /b 1
)
echo.
echo  [3/3] Done: valorant
echo.

echo ============================================================
echo  All 3 NanoDet re-training runs complete!
echo    universal_cloak\cf\%MODEL%\universal_noise.pt
echo    universal_cloak\overwatch\%MODEL%\universal_noise.pt
echo    universal_cloak\valorant\%MODEL%\universal_noise.pt
echo.
echo  Remember to close evaluation_summary.csv in Excel before
echo  running evaluations on these, or results won't save.
echo ============================================================
pause
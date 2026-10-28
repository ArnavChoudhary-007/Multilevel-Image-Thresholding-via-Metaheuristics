@echo off
REM Puts the new Week 2 work on its own branch, leaving main untouched.
REM Double-click this file. It can be deleted afterwards.
cd /d "%~dp0"

set "GIT=git"
where git >nul 2>nul
if errorlevel 1 (
    if exist "C:\Program Files\Git\cmd\git.exe" (
        set "GIT=C:\Program Files\Git\cmd\git.exe"
    ) else (
        echo Git was not found on this computer, so nothing was changed.
        echo Use the VS Code steps instead: click "main" bottom-left, then "Create new branch".
        pause
        exit /b 1
    )
)

echo Current branch:
"%GIT%" branch --show-current
echo.

"%GIT%" switch -c week2-dp-optimisers
if errorlevel 1 (
    echo.
    echo Could not create the branch, so nothing was committed. main is unchanged.
    pause
    exit /b 1
)

"%GIT%" add thresholding.py verify_week2.py measure_timing.py optimisers.py test_optimisers.py run_experiments.py landscape_k2.py analyse_stats.py sa_markov.py bitdepth_scaling.py make_figures.py PROGRESS_REPORT.md CLAUDE.md ASSP_Final_Report.docx ASSP_Final_Report.pdf
"%GIT%" commit -m "Exact DP, seven optimisers, full experiment, statistics, SA Markov and bit-depth analyses, final report" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_01SPzvqkvNgUoRgBa2kxnz3v"
if errorlevel 1 (
    echo.
    echo The commit failed. Your files are safe. Send this window's text to Claude.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Done. You are now on branch week2-dp-optimisers.
echo main has NOT been changed.
echo ============================================================
"%GIT%" log --oneline -3
echo.
echo Files not committed (should only be this script and results/figures):
"%GIT%" status --short
echo.
pause

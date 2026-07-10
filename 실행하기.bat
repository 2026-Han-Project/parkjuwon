@echo off
chcp 65001
cls
title AI 수요 예측 프로그램 설치 및 실행기

echo ========================================================
echo   AI 수요 예측 프로그램을 실행할 준비를 하고 있습니다.
echo   처음 실행 시에는 설치 시간이 조금 걸릴 수 있습니다.
echo   창을 닫지 말고 잠시만 기다려주세요!
echo ========================================================
echo.

:: 1. 가상환경이 없으면 생성
if not exist ".venv" (
    echo [1/3] 가상환경을 생성 중입니다...
    python -m venv .venv
)

:: 2. 가상환경 활성화 및 라이브러리 설치
echo [2/3] 필요한 기능을 설치 중입니다... (인터넷 연결 필요)
call .venv\Scripts\activate
pip install -r requirements.txt > nul 2>&1

:: 3. 프로그램 실행
echo.
echo [3/3] 프로그램을 실행합니다!
echo       잠시 후 인터넷 브라우저가 열립니다.
echo.
streamlit run app.py

pause
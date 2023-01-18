@ECHO OFF
CALL conda activate deploy
uvicorn docchat_app:app --reload
PAUSE
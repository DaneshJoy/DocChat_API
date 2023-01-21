@ECHO OFF
CALL conda activate deploy
uvicorn docchat_app:app --port 8080 --reload
PAUSE
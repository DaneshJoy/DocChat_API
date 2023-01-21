#!/bin/bash
# source /home/saeed/anaconda3/etc/profile.d/conda.sh
# conda activate deploy
uvicorn docchat_app:app --host 0.0.0.0 --port 8000 --reload
# gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080 docchat_app:app --workers 5

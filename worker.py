import os
import sys
from celery import Celery
from screenshot import Screenshot
import logging

broker_url = os.getenv("CELERY_BROKER_URL", "redis://redis-test:6379/0")

logging.info(f"REDIS URL: {broker_url}")

app = Celery(
    "screenshot_worker", 
    broker=broker_url, 
    backend=broker_url
)

model_path = os.getenv("MODEL_PATH")

if not model_path:
    print("MODEL_PATH not defined")
    sys.exit(1)

@app.task(name="screenshot")
def run_screenshot(input_path, output_dir):
    my_str = f"Got the task! Here's the video: {input_path}, outputting to {output_dir}"
    logging.info(my_str)
    ss = Screenshot("./model.pth", verbose=True)
    ss.screenshot(input_path, output_dir)

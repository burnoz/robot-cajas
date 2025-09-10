from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json
import os

app = FastAPI()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output.json")

@app.get("/robot-path")
def get_robot_path():
    if not os.path.exists(OUTPUT_PATH):
        return JSONResponse(content={"error": "output.json not found"}, status_code=404)
    with open(OUTPUT_PATH, "r") as f:
        data = json.load(f)
    return JSONResponse(content=data)

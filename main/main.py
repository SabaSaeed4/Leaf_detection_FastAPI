import uvicorn
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import RedirectResponse
from pydantic import BaseModel, validator
from serve_model import Predict, read_imagefile
from fastapi.middleware.cors import CORSMiddleware

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>
<h2>Try Tomato Leaf Detection api - it is just a learning app demo</h2>"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Tomato Leaf Detection", description=app_desc)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("JPG", "png", "jpg")
    logger.info(f"Received prediction request for file: {file.filename}")
    if not extension:
        logger.warning("Image must be jpg or png format!")
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    return Predict(image)


class Data(BaseModel):
    classs: str
    confidece_interval: str


@app.post("/")
def main(data: Data):
    logger.info(f"Received main request with data: {data}")
    return data.classs, data.confidece_interval


if __name__ == "__main__":
    uvicorn.run(app, debug=True)

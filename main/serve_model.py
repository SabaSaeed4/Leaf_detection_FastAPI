from io import BytesIO
import numpy as np
from PIL import Image
from keras.models import load_model

class_names = ["Apple", "Potato", "Tomato"]


def Load_model():
    model_dir = "D:\FYP\Fastapi\leaf_detection\model\leaf_detection.h5"
    model = load_model(model_dir, compile=False)
    print("Model loaded")
    return model


def Predict(image: Image.Image):
    model = None
    if model is None:
        model = Load_model()

    image = np.asarray(image.resize((256, 256)))[..., :3]
    image = np.expand_dims(image, 0)

    result = model.predict(image)
    pred = np.argmax(result, axis=1)

    response = []
    for res in result:
        resp = {
            "class": class_names[int(pred)],
            "confidence": f"{res[1] * 100:0.2f} %",
        }
        response.append(resp)

    return response


def read_imagefile(file) -> Image.Image:
    return Image.open(BytesIO(file))

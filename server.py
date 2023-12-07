import fastapi
import torchvision
import io
import base64
from number_drawer import draw_number
from image_drawer import draw_image
from torchvision.utils import save_image
from constants import MODE, Mode

app = fastapi.FastAPI()

@app.get("/")
async def hello():
    return "Hello, folks"

@app.post("/mnist_generator")
async def mnist_generator(request: fastapi.Request):
  text = (await request.json())["text"]
  if MODE == Mode.MNIST:
    img = draw_number(int(text))
  else:
    img = draw_image(int(text))
  buffer = io.BytesIO()
  save_image(img, buffer, format='PNG')
  #img.save(buffer, format="PNG")
  img_str = base64.b64encode(buffer.getvalue()).decode()
  img_str = f"data:image/png;base64,{img_str}"
  return { "img": img_str }

import fastapi
import torchvision
import io
import base64
import model

app = fastapi.FastAPI()

@app.get("/")
async def hello():
    return "Hello, folks"

@app.post("/mnist_generator")
async def mnist_generator(request: fastapi.Request):
  text = (await request.json())["text"]
  img = # do your magic :)
  img = torchvision.transforms.functional.to_pil_image(img)
  buffer = io.BytesIO()
  img.save(buffer, format="PNG")
  img_str = base64.b64encode(buffer.getvalue()).decode()
  img_str = f"data:image/png;base64,{img_str}"
  return { "img": img_str }

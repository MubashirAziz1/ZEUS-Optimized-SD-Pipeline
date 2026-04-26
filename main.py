import torch
from torchvision.utils import save_image

from zeus_sd_pipeline.zeus_pipeline import ZeusOptimizedStableDiffusionPipeline

pipe = ZeusOptimizedStableDiffusionPipeline.from_pretrained(
"stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
save_image(image, "output.png")
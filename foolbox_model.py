import foolbox
import numpy as np
import torchvision.models as models


MODEL_NAME = 'vgg16'

def create():
  model_fn = getattr(models, MODEL_NAME)
  model = model_fn(pretrained=True).eval()

  mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
  std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

  fmodel = foolbox.models.PyTorchModel(
    model,
    bounds=(0, 1),
    num_classes=1000,
    preprocessing=(mean, std))
  
  return fmodel


if __name__ == "__main__":
  print(create())

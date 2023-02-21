import torch
from model.model import model
from dataloader import getValDataloader
import torch

@torch.inference_mode() #@torch.inference_mode() is preferrable to @torch.no_grad() in all cases except when run time errors may occour.
def test(model,inputs,targets):
    testdataloader = getValDataloader(inputs,targets)
    for img,targ in iter(testdataloader):
        pred = model(img)
        loss_fn(pred,targ)

def loss_fn(pred,targ):
    pass

if __name__ == "__main__":
    inputs = ""
    targets = ""
    fakeModel = model()
    test(fakeModel,inputs,targets)
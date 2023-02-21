import torch.nn as nn # nn has loss functions can be used
# import torch

def loss_fn(preds,targets):
    return nn.CrossEntropyLoss()(preds,targets)

# #testing 
# x = torch.tensor([1,2,3,4]).float()
# y = torch.tensor([0.8,2.2,3.5,4])
# print(x.dtype,y.dtype)
# print(loss_fn(x.unsqueeze(0),y.unsqueeze(0)))
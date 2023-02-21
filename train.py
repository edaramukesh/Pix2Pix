# import torch
# from model.model import model
# from loss_functions import loss_fn
# from dataloader import getTrainDataloader,getValDataloader
# from torch.optim import Adam
# from torch.optim.lr_scheduler import LinearLR
# import wandb
# from tqdm import tqdm

# def train(model,optimizer:Adam,dataloader):
#     batch_losses = []
#     for inputs,targets in iter(dataloader):
#         optimizer.zero_grad()
#         preds = model(inputs)
#         loss = loss_fn(preds,targets)
#         batch_losses.append(loss.item())
#         loss.backward()
#         optimizer.step()
#     return sum(batch_losses)/len(batch_losses)

# @torch.inference_mode()
# def validation(model,dataloader):
#     batch_losses = []
#     for inputs,targets in iter(dataloader):
#         preds = model(inputs)
#         loss = loss_fn(preds,targets)
#         batch_losses.append(loss)
#     return sum(batch_losses)/len(batch_losses)

# def load_model(model:model,path):
#     state_dict = torch.load(path)
#     model.load_state_dict(state_dict)

# def mainFunc(model,optimizer,train_dataloader,val_dataloader,num_epochs):
#     wandb.init()
#     train_epoch_loss = 0
#     valid_epoch_loss = 0
#     for epoch_num in tqdm(range(1,num_epochs+1),desc="No of epochs"):
#         train_epoch_loss = train(model,optimizer,train_dataloader)
#         if epoch_num%2 == 0:
#             valid_epoch_loss = validation(model,val_dataloader)
#             wandb.log({"validation_epoch_loss":valid_epoch_loss})
#         if epoch_num%1 == 0:
#             wandb.log({"train_epoch_loss":train_epoch_loss})


# if __name__ == "__main__":
#     fakeModel = model()
#     pth_file_path = "checkpoints/model_v1.pth"
#     try:
#         load_model(fakeModel,pth_file_path)
#         print("loaded model succesfully")
#     except Exception:
#         print("Failed to load model")
#     learning_rate = 3e-4
#     tr_inputs = "training_data/images"
#     tr_targets = "training_data/masks"
#     vl_inputs = "validation_data/images"
#     vl_targets = "validation_data/masks"
#     optimizer = Adam(params=model.parameters(),lr=learning_rate)
#     train_dataloader = getTrainDataloader(tr_inputs,tr_targets)
#     val_dataloader = getValDataloader(vl_inputs,vl_targets)
#     num_epochs = 10
#     mainFunc(model=fakeModel,optimizer=optimizer,train_dataloader=train_dataloader,val_dataloader=val_dataloader,num_epochs=num_epochs)


from dataloader import get_loader
from model.model import Generator,Discriminator
from torch.optim import Adam
import torch.nn as nn
import torch
from tqdm import tqdm




train_path = "/home/mukesh/Desktop/4-2/cv_projects/pix2pix/train"
val_path = "/home/mukesh/Desktop/4-2/cv_projects/pix2pix/val"

train_dl = get_loader(train_path,train=True)
val_dl = get_loader(val_path,train=False)

g_model = Generator().cuda()
d_model = Discriminator(3).cuda()
g_lr = 2e-4
d_lr = 1e-5
g_optim = Adam(g_model.parameters(),g_lr,(0.5,0.999))
d_optim = Adam(d_model.parameters(),d_lr,(0.5,0.999))
loss_BCE = nn.BCEWithLogitsLoss()
loss_l1 =  nn.L1Loss()
g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()

def training(g_model,d_model,g_optim,d_optim,loss_BCE,loss_l1,g_scaler,d_scaler,train_dl,epoch):
  loop = tqdm(train_dl,leave=True)
  av_d_fake = []
  av_d_real = []
  for idx,(x,y) in enumerate(loop):
    x = x.cuda()
    y = y.cuda()

    with torch.cuda.amp.autocast():
      y_fake = g_model(x)
      D_real = d_model(x,y)
      D_real_loss = loss_BCE(D_real,torch.ones_like(D_real))
      D_fake = d_model(x,y_fake.detach())
      D_fake_loss = loss_BCE(D_fake,torch.zeros_like(D_fake))
      D_loss = (D_real_loss+D_fake_loss)/2

    if epoch%30==0:
      d_model.zero_grad()
      d_scaler.scale(D_loss).backward()
      d_scaler.step(d_optim)
      d_scaler.update()

    with torch.cuda.amp.autocast():
      D_fake = d_model(x, y_fake)
      G_fake_loss = loss_BCE(D_fake, torch.ones_like(D_fake))
      L1 = loss_l1(y_fake, y) * 100
      G_loss = G_fake_loss + L1

    g_optim.zero_grad()
    g_scaler.scale(G_loss).backward()
    g_scaler.step(g_optim)
    g_scaler.update()
    
    
    if idx % 10 == 0:
      av_d_fake.append(torch.sigmoid(D_fake).mean().item())
      av_d_real.append(torch.sigmoid(D_real).mean().item())
      loop.set_postfix( D_real=torch.sigmoid(D_real).mean().item(), D_fake=torch.sigmoid(D_fake).mean().item(),epoch_num = epoch)
  return sum(av_d_fake)/len(av_d_fake),sum(av_d_real)/len(av_d_real)

num_epochs = 100
d_model.train()
g_model.train()
try:
  with open("log1.txt","a")  as f:
    for epoch in range(num_epochs):
      D_fake,D_real = training(g_model,d_model,g_optim,d_optim,loss_BCE,loss_l1,g_scaler,d_scaler,train_dl,epoch)
      f.write(f"epoch_num={epoch}, D_fake={D_fake}, D_real={D_real}")
      f.write("\n")
except KeyboardInterrupt:
  checkpoint = {"g_model":g_model.state_dict(),"d_model":d_model.state_dict(),"opt_g":g_optim.state_dict(),
                "opt_d":d_optim.state_dict(),"g_sclr":g_scaler.state_dict(),"d_sclr":d_scaler.state_dict()}
  torch.save(checkpoint,"/home/mukesh/Desktop/4-2/cv_projects/pix2pix/checkpoints/checkpoint1.pth")

checkpoint = {"g_model":g_model.state_dict(),"d_model":d_model.state_dict(),"opt_g":g_optim.state_dict(),
                "opt_d":d_optim.state_dict(),"g_sclr":g_scaler.state_dict(),"d_sclr":d_scaler.state_dict()}
torch.save(checkpoint,"/home/mukesh/Desktop/4-2/cv_projects/pix2pix/checkpoints/checkpoint1.pth")
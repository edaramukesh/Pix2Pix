from torch.utils.data import DataLoader
# from dataset import CustomDataset
from dataset import custom_dataset

# def getTrainDataloader(inputs="training_data/images",targets="training_data/masks",phase="train"):
#     dataset = CustomDataset(inputs,targets,phase)
#     dataloader = DataLoader(dataset=dataset,shuffle=True)
#     return dataloader

# def getValDataloader(inputs="validation_data/images",targets="validation_data/masks",phase="val"):
#     dataset = CustomDataset(inputs,targets,phase)
#     dataloader = DataLoader(dataset=dataset,shuffle=False)
#     return dataloader

def get_loader(path,train):
  ds = custom_dataset(path,train)
  dl = DataLoader(ds,batch_size=1,shuffle=True,num_workers=4,pin_memory=True)
  return dl
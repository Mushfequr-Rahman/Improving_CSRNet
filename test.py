import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
from config import Config
from model import CSRNet
from dataset import *


def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    cfg= Config()
    device= cfg.device
    model=CSRNet()
    model.load_state_dict(torch.load(model_param_path))                           # GPU
    #torch.load(model_param_path, map_location=lambda storage, loc: storage)        # CPU
    model.to(device)
    """
    @Mushy 
    Changed data loader to give path From config device 
    
    """

    dataloader = create_test_dataloader(cfg.dataset_root)
    #dataloader=torch.utils.data.DataLoader(cfg.dataset_root,batch_size=1,shuffle=False)
    model.eval()
    mae=0
    with torch.no_grad():
        for i,data in enumerate(tqdm(dataloader)):
            """
            @Mushy 
            Changed how to access the data . 
            """

            img= data['image'].to(device)
            #gt_dmap=gt_dmap.to(device)
            gt_dmap = data['densitymap'].to(device)
            # forward propagation
            et_dmap=model(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader)))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    '''
    @Mushy
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cpu")
    model=CSRNet().to(device)
    model.load_state_dict(torch.load(model_param_path))                       # GPU
    #torch.load(model_param_path, map_location=lambda storage, loc: storage)    # CPU
    cfg = Config()
    dataloader = create_test_dataloader(cfg.dataset_root)
    model.eval()
    for i,data in enumerate(dataloader):
        if i==index:
            img = data['image'].to(device)
            # gt_dmap=gt_dmap.to(device)
            gt_dmap = data['densitymap'].to(device)
            # forward propagation
            et_dmap=model(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap,cmap=CM.gray)
            plt.show()
            break


if __name__=="__main__":
    torch.backends.cudnn.enabled=False

    img_root='./data/part_A_final/test_data/images'
    gt_dmap_root='./data/part_A_final/test_data/ground_truth'
    model_param_path='./checkpoints/shaghai_tech_a_best.pth'
    
    """
    img_root='./data/part_B_final/test_data/images'
    gt_dmap_root='./data/part_B_final/test_data/ground_truth'
    model_param_path='./checkpoints/Shanghai_Tech_B_1_best.pth'
    """
    cal_mae(img_root,gt_dmap_root,model_param_path)
    estimate_density_map(img_root,gt_dmap_root,model_param_path,3)
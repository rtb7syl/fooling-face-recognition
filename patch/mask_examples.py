import utils
from config import BaseConfiguration
import torch
from torchvision import transforms
from nn_modules import LandmarkExtractor, FaceXZooProjector
from PIL import Image
import os
from pathlib import Path

cfg = BaseConfiguration()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def apply_mask_to_imgs(datapath,save_path,maskpath):

    face_landmark_detector = utils.get_landmark_detector(cfg, device)
    location_extractor = LandmarkExtractor(device, face_landmark_detector, cfg.img_size).to(device)
    fxz_projector = FaceXZooProjector(device, cfg.img_size, cfg.patch_size).to(device)

    ids = os.listdir(datapath)
    print('No of ids ',len(ids))

    for id in ids:

        idpath = os.path.join(datapath,id)
        save_idpath=os.path.join(save_path,id)
        os.mkdir(save_idpath)
        img_names = os.listdir(idpath)

        for img_name in img_names:

            image_path=os.path.join(idpath, img_name)
            img_t = transforms.ToTensor()(Image.open(image_path)).unsqueeze(0).to(device)

            #transforms.ToPILImage()(img_t[0].cpu()).save(os.path.join('..', 'outputs', person_id, 'clean' + person_id + '.png'))

            mask_t = utils.load_mask(cfg, maskpath, device)
            uv_mask = mask_t[:, 3] if mask_t.shape[1] == 4 else None
            applied = utils.apply_mask(location_extractor, fxz_projector, img_t, mask_t[:, :3], uv_mask, is_3d=False)
            transforms.ToPILImage()(applied[0].cpu()).save(os.path.join(save_idpath, img_name))

if __name__ =="__main__":
    datapath='/home/lect0083/data/Umdfaces_subset_100_ids_test'
    save_path='/home/lect0083/data/Umdfaces_subset_100_ids_test_masked'
    maskpath='/home/lect0083/July/12-07-2022_15-50-42_28685121_sgd_lr_0.001_momentum/final_results/final_patch.png'
    apply_mask_to_imgs(datapath,save_path,maskpath)

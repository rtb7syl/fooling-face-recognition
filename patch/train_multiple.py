from config import patch_config_types
from train_impersonation import AdversarialMask
from test_impersonation import Evaluator
import os
from shutil import move
import pickle
from pathlib import Path
import datetime
import time
#from visualization import box_plots_examples
import torch
from torchvision import transforms
from PIL import Image


def train_multiple_persons():
    mode = 'targeted'
    config = patch_config_types[mode]()
    output_folders = []
    for i, lab in enumerate(sorted(os.listdir(config.train_img_dir)[:100])):
        config.update_current_dir()
        config.set_attribute('celeb_lab', [lab])
        config.set_attribute('celeb_lab_mapper', {0: lab})
        config.update_test_celeb_lab()
        adv_mask = AdversarialMask(config)
        print(f'Starting train person {i+1}...', flush=True)
        adv_mask.train()
        print('Finished train...', flush=True)
        evaluator = Evaluator(adv_mask)
        print(f'Starting test person {i+1}...', flush=True)
        evaluator.test()
        print('Finished test...', flush=True)
        output_folders.append(config.current_dir)
        del adv_mask, evaluator
        torch.cuda.empty_cache()

    print('Starting to create similarity boxes plots', flush=True)
    my_date = datetime.datetime.now()
    month_name = my_date.strftime("%B")
    final_output_path = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S") + '_' + os.environ['SLURM_JOBID'])
    Path(final_output_path).mkdir(parents=True, exist_ok=True)
    for output_folder in output_folders:
        move(output_folder, final_output_path)
    Path(os.path.join(final_output_path, 'combined_sim_boxes')).mkdir(parents=True, exist_ok=True)
    #box_plots_examples.gather_sim_and_plot(config.train_dataset_name, target_type='with_mask', embedder_names=config.test_embedder_names, job_id=os.environ['SLURM_JOBID'])
    #box_plots_examples.gather_sim_and_plot(config.train_dataset_name, target_type='without_mask', embedder_names=config.test_embedder_names, job_id=os.environ['SLURM_JOBID'])
    print('Finishing similarity boxes plots', flush=True)


def train_multiple_persons_impersonation():
    mode = 'targeted_impersonation'
    config = patch_config_types[mode]()
    output_folders = []
    for i, lab in enumerate(sorted(os.listdir(config.train_img_dir)[:config.train_number_of_people])):

        if lab==config.victim_celeb_lab[0]:
            continue

        config.update_current_dir()
        config.set_attribute('celeb_lab', [lab])
        config.set_attribute('celeb_lab_mapper', {0: lab})
        config.update_test_celeb_lab()
        adv_mask = AdversarialMask(config)
        print(f'Starting train person {i+1}...', flush=True)
        adv_mask.train()
        print('Finished train...', flush=True)
        adv_mask_t = adv_mask.best_patch
        evaluator = Evaluator(config,adv_mask_t)
        print(f'Starting test person {i+1}...', flush=True)
        evaluator.test()
        print('Finished test...', flush=True)
        output_folders.append(config.current_dir)
        del adv_mask, evaluator
        torch.cuda.empty_cache()

    print('Starting to create similarity boxes plots', flush=True)
    my_date = datetime.datetime.now()
    month_name = my_date.strftime("%B")
    final_output_path = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S") + '_' + os.environ['SLURM_JOBID'])
    Path(final_output_path).mkdir(parents=True, exist_ok=True)
    for output_folder in output_folders:
        move(output_folder, final_output_path)
    Path(os.path.join(final_output_path, 'combined_sim_boxes')).mkdir(parents=True, exist_ok=True)
    #box_plots_examples.gather_sim_and_plot(config.train_dataset_name, target_type='with_mask', embedder_names=config.test_embedder_names, job_id=os.environ['SLURM_JOBID'])
    #box_plots_examples.gather_sim_and_plot(config.train_dataset_name, target_type='without_mask', embedder_names=config.test_embedder_names, job_id=os.environ['SLURM_JOBID'])
    print('Finishing similarity boxes plots', flush=True)


if __name__ == '__main__':
    train_multiple_persons_impersonation()

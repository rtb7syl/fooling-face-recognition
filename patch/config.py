from torch import optim
import os
import datetime
import time


embedders_dict = {
    'resnet18': {
        'layers': [2, 2, 2, 2],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights', 'ms1mv3_arcface_resnet18.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights', 'glint360k_cosface_resnet18.pth')
            }
        }
    },
    'resnet34': {
        'layers': [3, 4, 6, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet34.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet34.pth')
            }
        }
    },
    'resnet50': {
        'layers': [3, 4, 14, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet50.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet50.pth')
            }
        }
    },
    'resnet100': {
        'layers': [3, 13, 30, 3],
        'heads': {
            'arcface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'ms1mv3_arcface_resnet100.pth')
            },
            'cosface': {
                'weights_path': os.path.join('..', 'face_recognition', 'insightface_torch', 'weights',
                                             'glint360k_cosface_resnet100.pth')
            },
            'magface': {
                'weights_path': os.path.join('..', 'face_recognition', 'magface_torch', 'weights',
                                             'magface_resnet100.pth')
            }
        }
    }
}


class BaseConfiguration:
    def __init__(self):
        self.seed = 42
        self.patch_name = 'base'

        # Train dataset options
        self.is_real_person = False
        self.train_dataset_name = 'CASIA-WebFace_aligned'
        self.train_img_dir = os.path.join('/home','lect0083','datasets', 'CASIA-WebFace_aligned')
        #self.train_img_dir = os.path.join('/home','lect0083','data', self.train_dataset_name, 'images')
        #self.train_img_dir = os.path.join('..','..', 'data', self.train_dataset_name)
        self.train_number_of_people = 100
        self.celeb_lab = os.listdir(self.train_img_dir)[:self.train_number_of_people]
        self.celeb_lab_mapper = {i: lab for i, lab in enumerate(self.celeb_lab)}
        self.num_of_train_images = 5

        self.shuffle = True
        self.img_size = (112, 112)
        self.train_batch_size = 4
        self.test_batch_size = 4
        self.magnification_ratio = 35

        # Attack options
        self.mask_aug = True
        self.patch_size = (112, 112)  # height, width
        self.initial_patch = 'white'  # body, white, random, stripes, l_stripes
        self.epochs = 100
        self.meta_optimizer='sgd'
        self.meta_lr=1e-3
        self.meta_momentum=0.9
        self.start_learning_rate = 1e-2
        self.es_patience = 7
        self.sc_patience = 2
        self.sc_min_lr = 1e-6
        self.scheduler_factory = lambda optimizer: optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                                        patience=self.sc_patience,
                                                                                        min_lr=self.sc_min_lr,
                                                                                        mode='min')

        # Landmark detection options
        self.landmark_detector_type = 'mobilefacenet'  # face_alignment, mobilefacenet

        # Embedder options
        #self.train_embedder_names = ['resnet100_arcface', 'resnet100_cosface']
        self.train_embedder_names = ['resnet100_arcface','resnet100_cosface','resnet18_arcface','resnet18_cosface']
        self.test_embedder_names = ['resnet100_arcface','resnet100_cosface','resnet18_arcface','resnet18_cosface','resnet34_arcface','resnet34_cosface','resnet50_arcface','resnet50_cosface']
        #self.test_embedder_names = ['resnet34_arcface','resnet34_cosface','resnet50_arcface','resnet50_cosface']



        # Loss options
        self.dist_loss_type = 'cossim'
        self.dist_weight = 1
        self.tv_weight = 0.1

        # Test options
        self.masks_path = os.path.join('..', 'data', 'masks')
        self.random_mask_path = os.path.join(self.masks_path, 'random.png')
        self.blue_mask_path = os.path.join(self.masks_path, 'blue.png')
        self.black_mask_path = os.path.join(self.masks_path, 'black.png')
        self.white_mask_path = os.path.join(self.masks_path, 'white.png')
        self.face1_mask_path = os.path.join(self.masks_path, 'face1.png')
        self.face3_mask_path = os.path.join(self.masks_path, 'face3.png')
        #self.current_dir='/home/lect0083/July/14-07-2022_03-37-34_28704568'
        self.update_current_dir()

    def set_attribute(self, name, value):
        setattr(self, name, value)

    def update_current_dir(self):
        my_date = datetime.datetime.now()
        month_name = my_date.strftime("%B")
        self.current_dir = os.path.join("experiments", month_name, time.strftime("%d-%m-%Y") + '_' + time.strftime("%H-%M-%S"))
        if 'SLURM_JOBID' in os.environ.keys():
            self.current_dir += '_' + os.environ['SLURM_JOBID']


class UniversalAttack(BaseConfiguration):
    def __init__(self):
        super(UniversalAttack, self).__init__()
        self.patch_name = 'universal'
        self.num_of_train_images = 5
        self.train_batch_size = 4
        self.test_batch_size = 32

        # Test dataset options
        self.test_num_of_images_for_emb = 5
        #self.test_dataset_names = ['CASIA-WebFace_aligned','Umdfaces']
        self.test_dataset_names = ['Umdfaces']
        #self.test_dataset_paths = [os.path.join('/home','lect0083','datasets', 'CASIA-WebFace_aligned'), os.path.join('/home','lect0083','data', 'Umdfaces', 'images')]
        self.test_img_dir = {name: os.path.join('/home','lect0083','data', 'Umdfaces', 'images') for name in self.test_dataset_names}
        #self.test_img_dir = {name: path for name, path in zip(self.test_dataset_names,self.test_dataset_paths)}
        print('self.test_img_dir',self.test_img_dir)
        self.test_number_of_people = 200
        self.test_celeb_lab = {}
        for dataset_name, img_dir in self.test_img_dir.items():
            label_list = os.listdir(img_dir)[:self.test_number_of_people]
            if dataset_name == self.train_dataset_name:
                label_list = os.listdir(img_dir)[-self.test_number_of_people:]
            self.test_celeb_lab[dataset_name] = label_list
        self.test_celeb_lab_mapper = {dataset_name: {i: lab for i, lab in enumerate(self.test_celeb_lab[dataset_name])}
                                      for dataset_name in self.test_dataset_names}


class TargetedAttack(BaseConfiguration):
    def __init__(self):
        super(TargetedAttack, self).__init__()
        self.patch_name = 'targeted'
        self.num_of_train_images = 10
        self.train_batch_size = 1
        self.test_batch_size = 4
        self.test_img_dir = {self.train_dataset_name: self.train_img_dir}

    def update_test_celeb_lab(self):
        self.test_celeb_lab = {self.train_dataset_name: self.celeb_lab}
        self.test_celeb_lab_mapper = {self.train_dataset_name: self.celeb_lab_mapper}

class UniversalImpersonationAttack(BaseConfiguration):
    def __init__(self,victim_id=None,lab=None):
        super(UniversalImpersonationAttack,self).__init__()

        self.patch_name="universal_impersonation"

        self.train_dataset_name = 'our_faces_and_Umdfaces_subset_100_ids_train'
        #self.train_img_dir='/home/ca550013/Projects/deep-learning-lab/data/our_faces_aligned/train'
        self.train_img_dir = os.path.join('/home','ca550013','Projects', 'deep-learning-lab', 'data', self.train_dataset_name)

        self.train_number_of_people = 102

        self.num_of_train_images = 5
        print('lab ', lab)

        if lab is None:
            self.victim_id=self.train_number_of_people if victim_id is None else victim_id
            lab = os.listdir(self.train_img_dir)[self.victim_id]
            print('lab', lab)

        self.celeb_lab = os.listdir(self.train_img_dir)
        self.celeb_lab.remove(lab)
        print('self.celeb_lab ',self.celeb_lab)
        self.celeb_lab_mapper = {i: lab for i, lab in enumerate(self.celeb_lab)}

        self.victim_celeb_lab = [lab]
            
        print('victim_celeb_lab ',self.victim_celeb_lab)
        self.victim_celeb_lab_mapper = {i: lab for i, lab in enumerate(self.victim_celeb_lab)}
        print('victim_celeb_lab_mapper ',self.victim_celeb_lab_mapper)
        self.num_of_victim_images=20

        self.dist_loss_type="inverse_cossim"
        self.tv_weight = 0.1

        #self.test_embedder_names = ['resnet34_arcface']

        self.test_batch_size = 32
        self.test_dataset_names = ['our_faces_aligned']
        self.test_img_dir = {name: os.path.join('/home','ca550013','Projects', 'deep-learning-lab', 'data', name, 'test') for name in self.test_dataset_names}
        self.test_number_of_people = 2

        self.test_celeb_lab = {}
        for dataset_name, img_dir in self.test_img_dir.items():
            #label_list = os.listdir(img_dir)[:self.test_number_of_people]
            label_list = os.listdir(img_dir)
            if dataset_name == self.train_dataset_name:
                #label_list = os.listdir(img_dir)[-self.test_number_of_people:]
                if self.victim_celeb_lab[0] in label_list:
                    label_list.remove(self.victim_celeb_lab[0])
            self.test_celeb_lab[dataset_name] = label_list
        self.test_celeb_lab_mapper = {dataset_name: {i: lab for i, lab in enumerate(self.test_celeb_lab[dataset_name])}
                                      for dataset_name in self.test_dataset_names}


class TargetedImpersonationAttack(BaseConfiguration):
    def __init__(self,victim_id=None):
        super(TargetedImpersonationAttack, self).__init__()
        self.patch_name = 'targeted_impersonation'

        #self.victim_id=self.train_number_of_people if victim_id is None else victim_id
        self.victim_id=100
        self.victim_celeb_lab = [os.listdir(self.train_img_dir)[self.victim_id]]
        print('victim_celeb_lab ',self.victim_celeb_lab)
        self.victim_celeb_lab_mapper = {i: lab for i, lab in enumerate(self.victim_celeb_lab)}
        print('victim_celeb_lab_mapper ',self.victim_celeb_lab_mapper)
        self.num_of_victim_images=10
        self.dist_loss_type="inverse_cossim"
        #self.tv_weight = 0.2
        
        self.train_number_of_people = 20
        self.num_of_train_images = 10
        self.train_batch_size = 1
        self.test_batch_size = 4
        self.test_img_dir = {self.train_dataset_name: self.train_img_dir}
        self.test_embedder_names = ['resnet34_arcface','resnet34_cosface','resnet50_arcface','resnet50_cosface']


    def update_test_celeb_lab(self):
        self.test_celeb_lab = {self.train_dataset_name: self.celeb_lab}
        self.test_celeb_lab_mapper = {self.train_dataset_name: self.celeb_lab_mapper}


patch_config_types = {
    "base": BaseConfiguration,
    "universal": UniversalAttack,
    "targeted": TargetedAttack,
    "universal_impersonation":UniversalImpersonationAttack,
    "targeted_impersonation":TargetedImpersonationAttack
}

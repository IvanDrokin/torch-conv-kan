import json
import os
from copy import deepcopy

import hydra
from omegaconf import OmegaConf, open_dict
import torch
import torch.nn as nn
from datasets import load_dataset
from imblearn.over_sampling import RandomOverSampler
from torchinfo import summary
from torchvision.transforms import v2



from kan_peft import PEFTVGGKAGN
from models import vggkagn, VGGKAGN
from train import Classification, train_model, FocalLoss


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def unpack_and_save_chunk(class_mapper, dataset, path2chunk, balance_data):
    os.makedirs(path2chunk, exist_ok=True)
    train_data = []
    for i in range(len(dataset)):
        sample = dataset[i]
        sample, label = sample['image'].convert('RGB'), sample['dx']
        label = class_mapper[label]
        sample.save(os.path.join(path2chunk, f'img_{i}.jpg'))
        train_data.append({'image': os.path.join(path2chunk, f'img_{i}.jpg'), 'label': label})
    if balance_data:
        train_data = oversample_data(train_data)
    return train_data


def oversample_data(data_dict):
    x = []
    y = []
    for i in range(len(data_dict)):
        x.append([i, ])
        y.append(data_dict[i]['label'])
    ros = RandomOverSampler(random_state=0, sampling_strategy='not majority')
    x_resampled, y_resampled = ros.fit_resample(x, y)
    output = []
    for ind in x_resampled:
        output.append(data_dict[ind[0]])
    return output


def check_and_load_chunck(class_mapper, dataset, cache_dir, pack_name, balance_data=False):
    if os.path.exists(os.path.join(cache_dir, '.'.join([pack_name, 'json']))):
        with open(os.path.join(cache_dir, '.'.join([pack_name, 'json'])), 'r') as f:
            _data = json.load(f)
    else:
        _data = unpack_and_save_chunk(class_mapper, dataset, os.path.join(cache_dir, pack_name), balance_data)
        with open(os.path.join(cache_dir, '.'.join([pack_name, 'json'])), 'w') as f:
            json.dump(_data, f)
    return _data


def unpack_imagenet(dataset, cache_dir='./data/skin_cancer_unpacked/'):
    class_mapper = {'actinic_keratoses': 0,
                    'dermatofibroma': 1,
                    'melanoma': 2,
                    'basal_cell_carcinoma': 3,
                    'melanocytic_Nevi': 4,
                    'vascular_lesions': 5,
                    'benign_keratosis-like_lesions': 6}

    print('READ TRAIN')
    train_data = check_and_load_chunck(class_mapper, dataset['train'], cache_dir, 'train', balance_data=True)
    print('READ VALIDATION')
    validation_data = check_and_load_chunck(class_mapper, dataset['validation'], cache_dir, 'validation')
    print('READ TEST')
    test_data = check_and_load_chunck(class_mapper, dataset['test'], cache_dir, 'test')
    return train_data, validation_data, test_data


def get_data(cfg):
    dataset = load_dataset("marmal88/skin_cancer", cache_dir='./data/skin_cancer/',
                           use_auth_token=True, trust_remote_code=True)

    if cfg.unpack_data:
        train_data, validation_data, test_data = unpack_imagenet(dataset, cache_dir='./data/skin_cancer_unpacked')
        del dataset
    else:
        train_data, validation_data, test_data = dataset['train'], dataset['validation'], dataset['test']

    transforms_train = v2.Compose([
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.RandomResizedCrop(224, antialias=True),
        # v2.RandomChoice([v2.AutoAugment(AutoAugmentPolicy.CIFAR10),
        #                  v2.AutoAugment(AutoAugmentPolicy.IMAGENET),
        #                  # v2.AutoAugment(AutoAugmentPolicy.SVHN),
        #                  # v2.TrivialAugmentWide()
        #                  ]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_val = v2.Compose([
        v2.ToImage(),
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = Classification(train_data, transform=transforms_train)
    # train_dataset = Classification(dataset['validation'], transform=transforms_train)
    val_dataset = Classification(validation_data, transform=transforms_val)
    test_dataset = Classification(test_data, transform=transforms_val)

    # num_grad_maps = 16
    #
    # samples_x = []
    # samples_x_pil = []
    # samples_y = []

    # layers = [0, 2, 5, 7, 10]
    # layers = cfg.visualization.layers
    # for i in range(num_grad_maps):
    #     sample, label = val_dataset.__getitem__(i)
    #     samples_x.append(sample)
    #     samples_y.append(label)
    #
    #     sample_norm = inverse_normalize(tensor=sample, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    #
    #     sample_norm = np.moveaxis(sample_norm.numpy() * 255, 0, -1).astype('uint8')
    #     samples_x_pil.append(Image.fromarray(sample_norm))
    # samples_x = torch.stack(samples_x, dim=0)
    # cam_reporter = GradCAMReporter(samples_x_pil, samples_x, samples_y, layers)
    cam_reporter = None
    return train_dataset, val_dataset, test_dataset, cam_reporter


def check_experiments(conducted_experiment, current_state):

    for exp in conducted_experiment:
        if exp[0] == current_state[0] and exp[1] == current_state[1] and exp[2] == current_state[2]:
            return True
    return False


@hydra.main(version_base=None, config_path="./configs/", config_name="skin_cancer.yaml")
def main(cfg):
    # model = moe_reskalnet_50x64p(3, 200, groups=cfg.model.groups,
    #                              degree=cfg.model.degree, width_scale=cfg.model.width_scale,
    #                              dropout=cfg.model.dropout, dropout_linear=cfg.model.dropout_linear)


    completed_experiments = []
    if os.path.exists('runned_exps_v2.json'):
        with open('runned_exps_v2.json', 'r') as f:
            completed_experiments = json.load(f)

    print(completed_experiments)

    original_cfg = deepcopy(cfg)
    head_types = ['Linear', ]
    # head_types = ['Linear', 'HiddenKAN', 'Matryoshka', 'KANtryoshka']
    # head_types = ['KANtryoshka', 'HiddenKAN']
    backbone_types = ['fixed', 'full', 'random', 'extra_degree', 'random_extra_degree',
                     'trainable 5', 'trainable 4, 5',
                     'trainable 4, 5, 6', 'trainable 5, 6', 'trainable 2', 'trainable 3', 'trainable 4']

    for head_type in head_types:
        for backbone_type in backbone_types:

            model_pretrained = vggkagn(3,
                                       1000,
                                       groups=1,
                                       degree=5,
                                       dropout=0.15,
                                       l1_decay=0,
                                       dropout_linear=0.25,
                                       width_scale=2,
                                       vgg_type='VGG11v2',
                                       expected_feature_shape=(1, 1),
                                       affine=True,
                                       )
            model_pretrained = model_pretrained.from_pretrained('brivangl/vgg_kagn11_v2')
            model = vggkagn(3,
                            7,
                            groups=1,
                            degree=5,
                            dropout=0.15,
                            l1_decay=0,
                            dropout_linear=0.25,
                            width_scale=2,
                            vgg_type='VGG11v2',
                            expected_feature_shape=(1, 1),
                            affine=True,
                            head_type=head_type
                            )
            if backbone_type != 'random' and backbone_type != 'random_extra_degree' and backbone_type != 'random_5':
                model.features.load_state_dict(model_pretrained.features.state_dict())
            model.avgpool = nn.AdaptiveMaxPool2d(model.expected_feature_shape)
            if backbone_type == 'fixed':

                if check_experiments(completed_experiments, [backbone_type, head_type, False]):
                    continue
                cfg = deepcopy(original_cfg)
                OmegaConf.set_struct(cfg, True)
                with open_dict(cfg):
                    cfg.peft_config = {'backbone_type': backbone_type, "head_type": head_type}

                    cfg.logging_dir = os.path.join(cfg.output_dir, f"{backbone_type}_{head_type}", 'train_logs')
                    cfg.output_dir = os.path.join(cfg.output_dir, f"{backbone_type}_{head_type}")

                    cfg.extra_table_log = {'backbone_type': backbone_type, "head_type": head_type, "finetune_base": False}

                model.features.requires_grad_(False)
                summary(model, (64, 3, 224, 224), device='cpu')
                dataset_train, dataset_val, dataset_test, cam_reporter = get_data(cfg)
                loss_func = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)

                train_model(model, dataset_train, dataset_val, loss_func, cfg,
                            dataset_test=dataset_test, cam_reporter=cam_reporter)
                completed_experiments.append((backbone_type, head_type, False))

                with open('runned_exps.json', 'w') as f:
                    json.dump(completed_experiments, f)
            elif backbone_type == 'full' or backbone_type == 'random':

                if check_experiments(completed_experiments, [backbone_type, head_type, True]):
                    continue
                cfg = deepcopy(original_cfg)
                OmegaConf.set_struct(cfg, True)
                with open_dict(cfg):
                    cfg.peft_config = {'backbone_type': backbone_type, "head_type": head_type}
                    cfg.logging_dir = os.path.join(cfg.output_dir, f"{backbone_type}_{head_type}", 'train_logs')
                    cfg.output_dir = os.path.join(cfg.output_dir, f"{backbone_type}_{head_type}")
                    cfg.extra_table_log = {'backbone_type': backbone_type, "head_type": head_type, "finetune_base": True}

                summary(model, (64, 3, 224, 224), device='cpu')
                dataset_train, dataset_val, dataset_test, cam_reporter = get_data(cfg)
                loss_func = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)

                train_model(model, dataset_train, dataset_val, loss_func, cfg,
                            dataset_test=dataset_test, cam_reporter=cam_reporter)
                completed_experiments.append((backbone_type, head_type, True))
                with open('runned_exps.json', 'w') as f:
                    json.dump(completed_experiments, f)
            else:
                for finetune_base in [True, False]:

                    if check_experiments(completed_experiments, [backbone_type, head_type, finetune_base]):
                        continue
                    if backbone_type == 'extra_degree' or backbone_type == 'random_extra_degree':
                        model = PEFTVGGKAGN(model, trainable_degrees=None, extra_degrees=1, finetune_base=finetune_base)
                    if backbone_type == 'random_5':
                        model = PEFTVGGKAGN(model, trainable_degrees=5, extra_degrees=0, finetune_base=finetune_base)
                    elif backbone_type == 'trainable 5':
                        model = PEFTVGGKAGN(model, trainable_degrees=[5, ], extra_degrees=0,
                                            finetune_base=finetune_base)
                    elif backbone_type == 'trainable 4, 5':
                        model = PEFTVGGKAGN(model, trainable_degrees=[4, 5], extra_degrees=0,
                                            finetune_base=finetune_base)
                    elif backbone_type == 'trainable 2':
                        model = PEFTVGGKAGN(model, trainable_degrees=[2, ], extra_degrees=1,
                                            finetune_base=finetune_base)
                    elif backbone_type == 'trainable 3':
                        model = PEFTVGGKAGN(model, trainable_degrees=[3, ], extra_degrees=1,
                                            finetune_base=finetune_base)
                    elif backbone_type == 'trainable 4':
                        model = PEFTVGGKAGN(model, trainable_degrees=[4, ], extra_degrees=1,
                                            finetune_base=finetune_base)
                    elif backbone_type == 'trainable 4, 5, 6':
                        model = PEFTVGGKAGN(model, trainable_degrees=[4, 5], extra_degrees=1,
                                            finetune_base=finetune_base)
                    elif backbone_type == 'trainable 5, 6':
                        model = PEFTVGGKAGN(model, trainable_degrees=[5, ], extra_degrees=1,
                                            finetune_base=finetune_base)

                    cfg = deepcopy(original_cfg)
                    OmegaConf.set_struct(cfg, True)
                    with open_dict(cfg):
                        cfg.peft_config = {'backbone_type': backbone_type, "head_type": head_type,
                                           'finetune_base': finetune_base}

                        cfg.extra_table_log = {'backbone_type': backbone_type, "head_type": head_type,
                                               "finetune_base": finetune_base}
                        cfg.logging_dir = os.path.join(cfg.output_dir, f"{backbone_type}_{head_type}_finetune_base{str(finetune_base).lower()}", 'train_logs')
                        cfg.output_dir = os.path.join(cfg.output_dir, f"{backbone_type}_{head_type}_finetune_base{str(finetune_base).lower()}")

                    summary(model, (64, 3, 224, 224), device='cpu')
                    dataset_train, dataset_val, dataset_test, cam_reporter = get_data(cfg)
                    # loss_func = nn.CrossEntropyLoss(label_smoothing=cfg.loss.label_smoothing)
                    loss_func = FocalLoss(gamma=1.5)


                    train_model(model, dataset_train, dataset_val, loss_func, cfg,
                                dataset_test=dataset_test, cam_reporter=cam_reporter)

                    completed_experiments.append((backbone_type, head_type, finetune_base))

                    with open('runned_exps_v2.json', 'w') as f:
                        json.dump(completed_experiments, f)


if __name__ == '__main__':
    main()
    # dataset = load_dataset("marmal88/skin_cancer", cache_dir='./data/skin_cancer', use_auth_token=True, trust_remote_code=True)
    # train_data, validation_data, test_data = unpack_imagenet(dataset, cache_dir='./data/skin_cancer_unpacked')


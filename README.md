# Slimmable Pruned Neural Networks (SP-Net)

Training and evaluation code of SP-Net on CIFAR10/100 and Imagenet classification tasks with pretrained models.

## Run

0. Requirements:
    * python3.8, pytorch 1.10, torchvision 0.11, albumentations, torch_ema.
    * CIFAR10/100 or ImageNet-1k datasets. For CIFAR10/100, you can download the datasets automatically just by executing this code. For ImageNet-1k, follow pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet).
1. Training, Fine-tuning, and Testing:
    * We use yaml config files under `apps` dir for training settings, and save trained models under `logs` dir. Each model is constructed based on files under `models` dir, and utility files are located under `utils` dir. The whole code is based on Pytorch.
    * Command:

          python train.py app:{apps/***.yml}
      `{apps/***.yml}` is config file. Do not miss `app:` prefix.
    * `{apps/default_setting.yml}` is config file for default settings. `{apps/s_***.yml}` for training S-Net. `{sp_***_train.yml}` for training SP-Net. `{sp_***_finetune.yml}` for fine-tuning SP-Net.
    * For fine-tuning, first train multiple base networks using `{sp_***_train.yml}`.
    * For training multiple base networks, you have to modify `{sp_***_train.yml}` and change `width_mult_list` in the config file so that the model is trained at the width. Note that you also have to change `best_model_path` and `checkpoint_path` in order not to overwrite existing trained model files. Models are trained with [DDP training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) by default. In order not to use multiple GPUs, use `CUDA_VISIBLE_DEVICES` environmental variable.
    * To test, uncomment `test_only` and `pretrained` in config file and run command.

## Pretrained models

### [SP-Net](https://arxiv.org/abs/2212.03415)

| Model | Params | FLOPs | Top-1 Err. | Files |
| :---: | ---: | ---: | :---: | :---: |
| SP-ResNet-50 | 25.6M<br>17.6M<br>10.3M | 4.1G<br>2.3G<br>1.1G | 23.5<br>24.0<br>25.9 | [98.1MB](https://drive.google.com/file/d/1gAuMiksD4ZrmvXky5zf6zLvB86_L3CQf/view?usp=share_link) |
| SP-VGGNet | 20.5M<br>6.6M<br>1.8M | 19.6G<br>4.9G<br>1.2G | 26.1<br>29.3<br>37.0 | [78.5MB](https://drive.google.com/file/d/1DZnxzAjx2BVO37BzkSoT5lGbAtPmpwXz/view?usp=share_link) |
| SP-MobileNetV1 | 4.9M<br>3.2M<br>1.8M | 569M<br>325M<br>150M | 27.3<br>28.9<br>32.2 | [33.6MB](https://drive.google.com/file/d/104F9W5u8-zobNFrFEqve5PHBq6w-sZHW/view?usp=share_link) |
| SP-MobileNetV2 | 5.4M<br>4.0M<br>2.9M | 509M<br>305M<br>207M | 25.2<br>26.5<br>27.7 | [21.1MB](https://drive.google.com/file/d/1xKGhanFG7HdrU9RktsCTp-rLvr3iuQKZ/view?usp=share_link) |

### Base models for pruning

| Model | Width | Params | FLOPs | Top-1 Err. | Files |
| :---: | :--- | ---: | ---: | :---: | :---: |
| ResNet-50 | ×1.0<br>×0.75<br>×0.5 | 25.6M<br>14.8M<br>6.9M | 4.1G<br>2.3G<br>1.1G | 23.5<br>24.7<br>27.8 | [97.8MB](https://drive.google.com/file/d/1-WLjpb93Koi6PSrTiX7kKuUm6q6HABSa/view?usp=share_link)<br>[56.6MB](https://drive.google.com/file/d/13_HBemITQNaLCh6fMQoVlF93qEcdlwV2/view?usp=share_link)<br>[26.6MB](https://drive.google.com/file/d/1dP8RnmMMK_t6gVfo6lsKZOFzQKOlKS1L/view?usp=share_link) |
| VGGNet | ×1.0<br>×0.75<br>×0.5 | 20.5M<br>11.7M<br>5.3M | 19.6G<br>11.0G<br>4.9G | 25.0<br>27.0<br>31.2 | [78.4MB](https://drive.google.com/file/d/1f197tt-Ul_Th6NF8b4CYird4xYzsta_T/view?usp=share_link)<br>[44.5MB](https://drive.google.com/file/d/1oHC_-y208bpZXKxVE0-tbpIycbG1gaT4/view?usp=share_link)<br>[20.1MB](https://drive.google.com/file/d/1gKpYeUDr9Thd24G3n5a4SyxxwycW66Bu/view?usp=share_link) |
| MobileNetV1 | ×1.3<br>×1.0<br>×0.75 | 6.7M<br>4.2M<br>2.6M | 946M<br>569M<br>325M | 26.3<br>27.7<br>29.9 | [25.8MB](https://drive.google.com/file/d/1rB06Q1liTqWCCelzs552HIB6lXQz8DNd/view?usp=share_link)<br>[16.3MB](https://drive.google.com/file/d/1QFePtJ1IfpVJwWWc6C7ABAXZ3MMl58aG/view?usp=share_link)<br>[10MB](https://drive.google.com/file/d/1WIHzr5dLziV_kDsZFYOBO_VOLM8yN66F/view?usp=share_link) |
| MobileNetV2 | ×1.3<br>×1.0 | 5.4M<br>3.5M | 509M<br>301M | 25.7<br>27.9 | [20.8MB](https://drive.google.com/file/d/1RazPV65nKwWhIB-r5dAI-RV_Xkr5LK1l/view?usp=share_link)<br>[13.6MB](https://drive.google.com/file/d/1-Qqg7-5whqmMXv7Rw1PbPRMaoDE6fSY7/view?usp=share_link) |

## Citation

If you find this code useful, please consider citing our work:
```tex
@article{kuratsu2022spnet,
  title={Slimmable Pruned Neural Networks},
  author={Kuratsu, Hideaki and Nakamura, Atsuyoshi},
  journal={arXiv preprint arXiv:2212.03415},
  year={2022}
}
```

## License

This code is based on Yu's [Slimmable Neural Networks](https://github.com/JiahuiYu/slimmable_networks), and released under the [MIT License](./LICENSE).

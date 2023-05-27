# ResNet18, ResNet18-FCN32, U-Net

Part 1 - [ResNet18 for classification](https://colab.research.google.com/drive/1RipCF-PXzo5c8tA4fxmzmz0c8kuXxnTi?usp=sharing). Over 90% CA achieved after training for 15 epochs with Adam, LR 0.001, batch size 64, and [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html) with `step_size = 10`, `gamma = 0.1`.

Part 2 - [ResNet18-FCN32 and UNet for image segmentation](https://colab.research.google.com/drive/1ZhHRxQOnhhoXyJsGXja2bBVKJo_uja-I?usp=sharing). Both networks were trained for 5 epochs with Adam, with batch size 4, and LR set to 0.001.

Part 3 - [UNet for image colorization](https://colab.research.google.com/drive/1AEd_YNuo0LBlJp-Vdm3tYFBU_bizimJa?usp=sharing). Network was trained for 5 epochs with Adam, batch size 16 and LR set to 0.001. Skip connections were used.

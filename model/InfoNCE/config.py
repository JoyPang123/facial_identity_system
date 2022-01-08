import torch


class Config:
    # Path settings
    csv_path = "/home/coursome123/celebA.csv"
    img_root = "/home/coursome123/img_align_celeba"
    
    # Model settings
    model_type = "resnet18"
    out_dim = 128
    pretrained = True

    # Training device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyper parameters
    size = 224
    epochs = 200
    batch_size = 1024
    lr = 3e-4
    weight_decay = 1e-4
    fp16_precision = True
    temperature = 0.7
    n_views = 2


config = Config()

from data import ReviewDataset, collate_fn
from model import Model
from config import config
from train import train
from utils import get_n_parameters

from torch.utils.data import DataLoader, random_split


def main():
    dataset = ReviewDataset(train=True)
    train_dataset, eval_dataset = random_split(dataset, lengths=[0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"],
                              drop_last=True, shuffle=False,
                              collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"],
                             drop_last=True, shuffle=True,
                             collate_fn=collate_fn)

    model = Model().cuda()
    get_n_parameters(model)
    train(model, train_loader, eval_loader)


if __name__=="__main__":
    main()

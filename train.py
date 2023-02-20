import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from sklearn.metrics import f1_score
from tqdm import tqdm
import wandb

from config import config


def custom_loss():
    # Take into account ordinal information regarding the ratings with MSE loss
    pass


def train(model, train_loader, eval_loader):
    optimizer = Adam(model.parameters(), lr=config["lr"])
    if config["use_weights"]:
        weights = train_loader.dataset.dataset.weights
        criterion = CrossEntropyLoss(weights)
    else:
        criterion = CrossEntropyLoss()

    wandb.init(project="goodreads_book_reviews", config=config)
    wandb.watch(model, criterion, log_freq=len(train_loader) // 10)
    tot = len(train_loader)

    for epoch in tqdm(range(config["epochs"]), desc="Epoch"):

        for idx, data in tqdm(enumerate(train_loader), leave=False, total=tot):
            
            input_id = data[0]
            attention_mask = data[1]
            eval_target = data[2]

            model.train()
            optimizer.zero_grad()
            train_output = model(input_id, attention_mask)
            train_loss = criterion(train_output, eval_target.argmax(dim=1))
            train_loss.backward()
            optimizer.step()
            
            if idx % 10 == 0:
                eval_metrics = validation_step(model, criterion, eval_loader)

            wandb.log({
                "Train loss": train_loss,
                "Eval loss": eval_metrics[0],
                "Eval f1": eval_metrics[1],
                "Eval acc": eval_metrics[2]
            }, step=idx)

    torch.save(model.state_dict(), "model/temp.pt")


def validation_step(model, criterion, eval_loader):
    idx, (input_id, attention_mask, eval_target) = next(enumerate(eval_loader))
    model.eval()
    eval_output = model(input_id, attention_mask)
    eval_loss = criterion(eval_output, eval_target.argmax(dim=1))

    eval_output_argmax = eval_output.argmax(dim=1).detach().cpu()
    eval_target_argmax = eval_target.argmax(dim=1).detach().cpu()

    # Calculate metrics
    eval_acc = (eval_output_argmax == eval_target_argmax).sum().item() \
                   / config["batch_size"]
    eval_f1 = f1_score(eval_target_argmax.tolist(), eval_output_argmax.tolist(), 
                       average="micro")

    return eval_loss, eval_f1, eval_acc

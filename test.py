import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from data import ReviewDataset, collate_fn
from model import Model
from config import config


# Load data
dataset = ReviewDataset(train=False)
test_loader = DataLoader(dataset, batch_size=200, drop_last=False,
                         shuffle=False, collate_fn=collate_fn)
review_id = dataset.review_id.tolist()
prediction = []

# Load model
model = Model()
model.load_state_dict(torch.load("model/rural-oath-45.pt"))
model.cuda().eval()

# Make predictions
for idx, (input_id, attention_mask, variables) in tqdm(enumerate(test_loader), total=len(test_loader)):
    output = model(input_id, attention_mask, variables).argmax(dim=1).detach().cpu().tolist()
    prediction.extend(output)

# Save the predictions for submission
with open("data/prediction_rural-oath-45.csv", "w") as file:
    file = csv.writer(file, delimiter=",")
    file.writerow(["review_id", "rating"])
    for i, j in zip(review_id, prediction):
        file.writerow([i, j])

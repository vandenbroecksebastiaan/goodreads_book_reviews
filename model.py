from torch import nn
from transformers import AutoModel

from config import config


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(
            config["model_name"], output_hidden_states=True
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        self.bn1 = nn.BatchNorm1d(num_features=768*config["max_length"])
        self.fc1 = nn.Linear(768*config["max_length"], config["hidden_size"])

        self.bn2 = nn.BatchNorm1d(config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], 6)

    def forward(self, input_id, attention_mask):
        distilbert_output = self.pretrained_model(input_id, attention_mask)
        last_hidden_state = distilbert_output.last_hidden_state
        last_hidden_state = last_hidden_state.permute((1, 0, 2))
        last_hidden_state = last_hidden_state.flatten(start_dim=1, end_dim=2)

        output = self.dropout(last_hidden_state)
        output = self.relu(last_hidden_state)
        output = self.bn1(output)
        output = self.fc1(output)

        output = self.dropout(output)
        output = self.relu(output)
        output = self.bn2(output)

        return self.fc2(output)

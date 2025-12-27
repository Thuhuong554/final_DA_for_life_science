import torch
import torch.nn as nn
from transformers import AutoModel

class ChemBERTaReference(nn.Module):
    """
    ChemBERTa + Linear classification head
    """
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
        logits = self.classifier(cls_emb).squeeze(-1)
        return logits


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM model for bioactivity prediction
    """
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64, dropout_rate=0.5):
        super(HybridCNNLSTM, self).__init__()
        
        # Embedding: Transforms integers into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # CNN Block: Extract Local Features
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)  # Helps with faster convergence
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM Block: Learn sequence dependencies
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True, bidirectional=True)
        
        # Classification Head
        self.dropout = nn.Dropout(dropout_rate)  # avoid Overfit
        self.fc1 = nn.Linear(hidden_dim * 2, 64) 
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embedding(x).permute(0, 2, 1)  # Change dimension for CNN: [batch, embed, seq]
        
        # CNN Flow
        c = self.conv1(emb)
        c = self.bn1(c)
        c = self.relu(c)
        c = self.pool(c)  # [batch, 64, seq_len/2]
        
        # LSTM Flow
        l_in = c.permute(0, 2, 1)  # [batch, seq, feat]
        out, (hn, _) = self.lstm(l_in)
        
        # get final hidden state (combine both directions)
        final_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        
        # Fully Connected
        x = self.fc1(final_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)
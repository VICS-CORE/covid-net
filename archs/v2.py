import torch
tnn = torch.nn


class CovidNet(tnn.Module):
    def __init__(self, ip_seq_len=1, op_seq_len=1, ip_size=1, op_size=1, hidden_size=1, num_layers=1, dropout=0.5, **kwargs):
        super(CovidNet, self).__init__()
        
        self.ip_seq_len = ip_seq_len
        self.op_seq_len = op_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ip_size = ip_size
        self.op_size = op_size
        self.dropout = dropout
        
        self.lstm = tnn.LSTM(
            input_size=self.ip_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dropout = tnn.Dropout(p=self.dropout)
        self.linear = tnn.Linear(self.hidden_size * self.ip_seq_len, self.op_size * self.op_seq_len)
        self.sigmoid = tnn.Sigmoid()
    
    def forward(self, ip, **kwargs):
        lstm_out, _ = self.lstm(ip)
        dropout_out = self.dropout(lstm_out.reshape(-1, self.hidden_size * self.ip_seq_len))
        linear_out = self.linear(dropout_out)
        sigmoid_out = self.sigmoid(linear_out.view(-1, self.op_seq_len, self.op_size))
        return sigmoid_out
    
    def predict(self, ip, **kwargs):
        with torch.no_grad():
            preds = self.forward(ip)
        return preds

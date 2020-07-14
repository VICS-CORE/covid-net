import torch
tnn = torch.nn


class CovidNet(tnn.Module):
    def __init__(self, ip_seq_len=1, op_seq_len=1, ip_size=1, op_size=1, ip_aux_size=0, hidden_size=1, num_layers=1, dropout=0.5):
        super(CovidNet, self).__init__()
        
        self.ip_seq_len = ip_seq_len
        self.op_seq_len = op_seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ip_size = ip_size
        self.op_size = op_size
        self.dropout = dropout
        self.ip_aux_size = ip_aux_size
        
        self.lstm = tnn.LSTM(
            input_size=self.ip_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.dropout = tnn.Dropout(p=self.dropout)
        self.linear = tnn.Linear(self.hidden_size * self.ip_seq_len, self.op_size * self.op_seq_len)
        self.sigmoid = tnn.Sigmoid()
        # if aux input present, initialise cell state with it
        if self.ip_aux_size:
            self.aux = tnn.Linear(self.ip_aux_size, self.num_layers * self.hidden_size)
    
    def forward(self, ip, aux_ip=None):
        batch_size = ip.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        if self.ip_aux_size:
            c0 = self.aux(aux_ip.view(-1, self.ip_aux_size)).view(self.num_layers, batch_size, self.hidden_size)
        else:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        lstm_out, _ = self.lstm(ip, (h0, c0))
        dropout_out = self.dropout(lstm_out.reshape(-1, self.hidden_size * self.ip_seq_len))
        linear_out = self.linear(dropout_out)
        sigmoid_out = self.sigmoid(linear_out.view(-1, self.op_seq_len, self.op_size))
        return sigmoid_out
    
    def predict(self, ip, aux_ip=None):
        with torch.no_grad():
            preds = self.forward(ip, aux_ip)
        return preds

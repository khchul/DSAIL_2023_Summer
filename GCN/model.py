import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    '''
    A_ : D^(-1/2) * A * D^(-1/2)
    input_ch : Dimension of node feature vectors
    class_num : Number of classes
    f : Dimension of weight vectors
    Dropout : Whether to dropout weights
    '''
    def __init__(self, A_, input_ch, class_num, args):
        super(GCN, self).__init__()

        self.A_ = A_
        self.input_ch = input_ch
        self.class_num = class_num
        self.dropout = args.dropout

        self.fc1 = nn.Linear(input_ch, args.F)
        self.fc2 = nn.Linear(args.F, class_num)
        if self.dropout:
            self.do = nn.Dropout(args.dropout_rate)

    def forward(self, X):
        output = torch.mm(self.A_, self.fc1(X))
        ouput = F.relu(output)
        
        if self.dropout:
            output = self.do(output)
        
        output = torch.mm(self.A_, self.fc2(output))
        
        return output

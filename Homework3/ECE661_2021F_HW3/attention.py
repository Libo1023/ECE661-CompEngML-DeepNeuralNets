import torch
import torch.nn as nn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    """
    Q: Implement the MultiHeadAttention's forward function. 
    Use the layers initialized in the __init__ function.
    Compute attention between query and key. 
    Apply the obtained attention scores on value.  
    Take care of the multi-head machenism. 
    Apply a fc layer to the output before return.
    """
    def forward(self, query, key, value, mask=None):
        """
        Forward function
        :param query: [batch size, sequence length, hidden dim]
        :param key: [batch size, sequence length, hidden dim]
        :param value: [batch size, sequence length, hidden dim]
        :param mask: Just pass None to mask. No need to handle it specifically.
        :return: [batch size, sequence length, number of heads, hidden dim]
        """
        # Add your code here.
        # pass  # placeholder

        # Extract the value of batch size, 
        # sequence length, and
        # hidden dim respectively
        batch_size      = value.size(0)
        sequence_length = value.size(1)
        hidden_dim      = value.size(2)

        # Transpose query, key, and value into the shape of 
        # [sequence length, batch size, hidden dim]
        query_sbh = query.permute(1, 0, 2)
        key_sbh     = key.permute(1, 0, 2)
        value_sbh = value.permute(1, 0, 2)

        fc_query = self.linear_layers[0](query_sbh)
        fc_key   = self.linear_layers[1](key_sbh)
        fc_value = self.linear_layers[2](value_sbh)

        query_dim = fc_query.view(sequence_length, batch_size, self.h, self.d_k)
        key_dim     = fc_key.view(sequence_length, batch_size, self.h, self.d_k)
        value_dim = fc_value.view(sequence_length, batch_size, self.h, self.d_k)

        # Transpose query, key, and value into the shape of 
        # [batch_size, self.h, sequence_length, self.d_k]
        # for matrix multiplication
        query_matrix = query_dim.permute(1, 2, 0, 3)
        key_matrix     = key_dim.permute(1, 2, 0, 3)
        value_matrix = value_dim.permute(1, 2, 0, 3)

        attention_scores = torch.matmul(query_matrix, key_matrix.permute(0, 1, 3, 2))
        # Scale the attention_scores
        attention_scores = attention_scores / float(self.d_k ** (0.5))
        # Apply softmax function to attention_scores
        # to attain the attention_weights
        attention_weights = torch.softmax(attention_scores, dim = -1)

        dropout_weights = self.dropout(attention_weights)
        out = torch.matmul(dropout_weights, value_matrix)

        # Transpose out matrix into the shape of 
        # [sequence_length, batch_size, self.h, self.d_k]
        # for concatenation
        out = out.permute(2, 0, 1, 3)
        out = out.contiguous()
        # Concatenation
        out = out.view(sequence_length, batch_size, hidden_dim)

        # Transpose out matrix into the shape of 
        # [batch_size, sequence_length, hidden_dim]
        # for final return
        out = out.permute(1, 0, 2)

        # Apply a fc layer to the output before return
        out = self.output_linear(out)

        return out


if __name__ == '__main__':
    # test the module.
    pass

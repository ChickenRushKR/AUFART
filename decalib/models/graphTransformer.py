import torch
import torch.nn as nn
import torch.nn.functional as F
from OpenGraphAU.model.graph import create_e_matrix
from OpenGraphAU.model.basic_block import *

# Gated GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNNLayer(nn.Module):
    def __init__(self, in_channels, num_classes, dropout_rate = 0.1):
        super(GNNLayer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        dim_in = self.in_channels
        dim_out = self.in_channels

        self.U = nn.Linear(dim_in, dim_out, bias=False)
        self.V = nn.Linear(dim_in, dim_out, bias=False)
        self.A = nn.Linear(dim_in, dim_out, bias=False)
        self.B = nn.Linear(dim_in, dim_out, bias=False)
        self.E = nn.Linear(dim_in, dim_out, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(2)

        self.bnv = nn.BatchNorm1d(num_classes)
        self.bne = nn.BatchNorm1d(num_classes * num_classes)

        self.act = nn.ReLU()

        self.init_weights_linear(dim_in, 1)


    def init_weights_linear(self, dim_in, gain):
        # conv1
        scale = gain * np.sqrt(2.0 / dim_in)
        self.U.weight.data.normal_(0, scale)
        self.V.weight.data.normal_(0, scale)
        self.A.weight.data.normal_(0, scale)
        self.B.weight.data.normal_(0, scale)
        self.E.weight.data.normal_(0, scale)


        bn_init(self.bnv)
        bn_init(self.bne)


    def forward(self, x, edge, start, end):

        res = x
        Vix = self.A(x)  # V x d_out
        Vjx = self.B(x)  # V x d_out
        e = self.E(edge)  # E x d_out
        # print(e.shape)
        # print(x.shape)
        # print(start.shape)
        # print(end.shape)

        edge = edge + self.act(self.bne(torch.einsum('ev, bvc -> bec', (end, Vix)) + torch.einsum('ev, bvc -> bec',(start, Vjx)) + e))  # E x d_out

        e = self.sigmoid(edge)
        b, _, c = e.shape
        e = e.view(b,self.num_classes, self.num_classes, c)
        e = self.softmax(e)
        e = e.view(b, -1, c)


        Ujx = self.V(x)  # V x H_out
        Ujx = torch.einsum('ev, bvc -> bec', (start, Ujx))  # E x H_out
        Uix = self.U(x)  # V x H_out
        x = Uix + torch.einsum('ve, bec -> bvc', (end.t(), e * Ujx)) / self.num_classes  # V x H_out
        x = res + self.act(self.bnv(x))

        return x, edge



# GAT GCN Used to Learn Multi-dimensional Edge Features and Node Features
class GNN(nn.Module):
    def __init__(self, in_channels, num_classes, layer_num = 2):
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        start, end = create_e_matrix(self.num_classes)
        self.start = Variable(start, requires_grad=False)
        self.end = Variable(end, requires_grad=False)

        graph_layers = []
        for i in range(layer_num):
            layer = GNNLayer(self.in_channels, self.num_classes)
            graph_layers += [layer]

        self.graph_layers = nn.ModuleList(graph_layers)


    def forward(self, x, edge):
        dev = x.get_device()
        if dev >= 0:
            self.start = self.start.to(dev)
            self.end = self.end.to(dev)
        for i, layer in enumerate(self.graph_layers):
            x, edge = layer(x, edge, self.start, self.end)
        return x, edge


class GraphTransformer(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, output_dim, num_heads=4, hidden_dim=256):
        super(GraphTransformer, self).__init__()
        self.in_channels=512
        self.num_main_classes=27
        self.gnn = GNN(self.in_channels, self.num_main_classes, 2)

        # Node and Edge embeddings
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)
        
        # Transformer Decoder layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim,
            )
            for _ in range(4)  # You can adjust the number of layers as needed
        ])
        # self.transformer_layer = nn.TransformerDecoderLayer(
                # d_model=hidden_dim,
                # nhead=num_heads,
                # dim_feedforward=hidden_dim,
            # )
        # MLP Head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, node, edge):

        node, edge = self.gnn(node, edge)
        # Embed node and edge features
        node_embedded = self.node_embedding(node)  # Shape: (batch_size, num_nodes, hidden_dim)
        edge_embedded = self.edge_embedding(edge)  # Shape: (batch_size, num_nodes, num_nodes, hidden_dim)
        
        # Reshape edge_embedded for Transformer input
        # batch_size, num_square_nodes, hidden_dim = _embedded.size()
        # edge_embedded = edge_embedded.view(batch_size, -1, hidden_dim)  # Shape: (batch_size, num_nodes*num_nodes, hidden_dim)
        
        # Transformer Decoder

        # node_embedded = self.transformer_layer(node_embedded, edge_embedded)
        for layer in self.transformer_layers:
            # node_embedded = node_embedded.permute(1,0,2)
            # edge_embedded = edge_embedded.permute(1,0,2)
            # node_embedded = layer(node_embedded, src_key_padding_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None, memory=None, src_mask=None, tgt_mask=None, edge_embedded=edge_embedded)
            node_embedded = layer(node_embedded, edge_embedded, memory_key_padding_mask=None, tgt_key_padding_mask=None)
            # node_embedded = node_embedded.permute(1,0,2)
        
        # MLP Head for final prediction
        output = self.mlp(node_embedded)  # Shape: (batch_size, num_nodes, output_dim)
        
        return output


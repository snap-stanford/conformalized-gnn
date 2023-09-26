import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, SGConv

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, backbone, heads = 1, aggr = 'sum'):
        super().__init__()
        if backbone == 'GCN':
            self.conv1 = GCNConv(in_channels, hidden_channels, cached=True,
                             normalize=True)
            self.conv2 = GCNConv(hidden_channels, out_channels, cached=True,
                             normalize=True)
        elif backbone == 'GAT':
            self.conv1 = GATConv(in_channels, hidden_channels, heads)
            self.conv2 = GATConv(hidden_channels, out_channels, heads)
        elif backbone == 'GraphSAGE':
            self.conv1 = SAGEConv(in_channels, hidden_channels, aggr)
            self.conv2 = SAGEConv(hidden_channels, out_channels, aggr)
        elif backbone == 'SGC':
            self.conv1 = SGConv(in_channels, hidden_channels)
            self.conv2 = SGConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None, return_h = False):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        if return_h:
            return x
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
class GNN_Multi_Layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, backbone, heads = 1, aggr = 'sum', num_layers = 2):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        if num_layers == 1:
            if backbone == 'GCN':
                self.convs.append(GCNConv(in_channels, out_channels, cached=True, normalize=True))
            elif backbone == 'GAT':
                self.convs.append(GATConv(in_channels, out_channels, heads))
            elif backbone == 'GraphSAGE':
                self.convs.append(SAGEConv(in_channels, out_channels, aggr))
            elif backbone == 'SGC':
                self.convs.append(SGConv(in_channels, out_channels))
        else:
            if backbone == 'GCN':
                self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, normalize=True))
            elif backbone == 'GAT':
                self.convs.append(GATConv(in_channels, hidden_channels, heads))
            elif backbone == 'GraphSAGE':
                self.convs.append(SAGEConv(in_channels, hidden_channels, aggr))
            elif backbone == 'SGC':
                self.convs.append(SGConv(in_channels, hidden_channels))
            for _ in range(num_layers-2):
                if backbone == 'GCN':
                    self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True, normalize=True))
                elif backbone == 'GAT':
                    self.convs.append(GATConv(hidden_channels, hidden_channels, heads))
                elif backbone == 'GraphSAGE':
                    self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr))
                elif backbone == 'SGC':
                    self.convs.append(SGConv(hidden_channels, hidden_channels))
            if backbone == 'GCN':
                self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, normalize=True))
            elif backbone == 'GAT':
                self.convs.append(GATConv(hidden_channels, out_channels, heads))
            elif backbone == 'GraphSAGE':
                self.convs.append(SAGEConv(hidden_channels, out_channels, aggr))
            elif backbone == 'SGC':
                self.convs.append(SGConv(hidden_channels, out_channels))
                
    def forward(self, x, edge_index, edge_weight=None):
        for idx, conv in enumerate(self.convs):
            x = F.dropout(x, p=0.5, training=self.training)
            if idx == len(self.convs) - 1:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index, edge_weight).relu()
        return x
    
    
class SimpleMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.FC_hidden = nn.Linear(input_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.ReLU = nn.ReLU()       
                
    def forward(self, x):
        h     = self.ReLU(self.FC_hidden(x))
        h     = self.ReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat
    
    
class ConfGNN(torch.nn.Module):
    def __init__(self, model, dataset, args, num_conf_layers, base_model, output_dim, task):
        super().__init__()
        self.model = model
        #num_classes = max(dataset.y).item() + 1
        #print(base_model)
        self.confgnn = GNN_Multi_Layer(output_dim, args.confnn_hidden_dim, output_dim, base_model, args.heads, args.aggr, num_conf_layers)  
        self.task = task
    def forward(self, x, edge_index):
        with torch.no_grad():
            scores = self.model(x, edge_index)
        if self.task == 'regression':
            out = scores
        else:
            out = F.softmax(scores, dim = 1)
        adjust_scores = self.confgnn(out, edge_index)
        return adjust_scores, scores

    
class ConfMLP(torch.nn.Module):
    def __init__(self, model, dataset, output_dim, task):
        super().__init__()
        self.model = model
        self.confmlp = SimpleMLP(output_dim, 64, output_dim)  
        self.task = task
        
    def forward(self, x, edge_index):
    
        with torch.no_grad():
            scores = self.model(x, edge_index)
        if self.task == 'regression':
            out = scores
        else:
            out = F.softmax(scores, dim = 1)
        adjust_scores = self.confmlp(out)
        return adjust_scores, scores
    
    
def fit_calibration(temp_model, eval, data, train_mask, test_mask, patience = 100):
    """
    Train calibrator
    """    
    vlss_mn = float('Inf')
    with torch.no_grad():
        logits = temp_model.model(data.x, data.edge_index)
        labels = data.y
        edge_index = data.edge_index
        model_dict = temp_model.state_dict()
        parameters = {k: v for k,v in model_dict.items() if k.split(".")[0] != "model"}
    for epoch in range(2000):
        temp_model.optimizer.zero_grad()
        temp_model.train()
        # Post-hoc calibration set the classifier to the evaluation mode
        temp_model.model.eval()
        assert not temp_model.model.training
        calibrated = eval(logits)
        loss = F.cross_entropy(calibrated[train_mask], labels[train_mask])
        # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
        # margin_reg = 0.
        # loss = loss + margin_reg * dist_reg
        loss.backward()
        temp_model.optimizer.step()

        with torch.no_grad():
            temp_model.eval()
            calibrated = eval(logits)
            val_loss = F.cross_entropy(calibrated[test_mask], labels[test_mask])
            # dist_reg = intra_distance_loss(calibrated[train_mask], labels[train_mask])
            # val_loss = val_loss + margin_reg * dist_reg
            if val_loss <= vlss_mn:
                state_dict_early_model = copy.deepcopy(parameters)
                vlss_mn = np.min((val_loss.cpu().numpy(), vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step >= patience:
                    break
    model_dict.update(state_dict_early_model)
    temp_model.load_state_dict(model_dict)
    
    
class TS(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))
        self.device = device
        
    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.temperature_scale(logits)
        return logits / temperature

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(self.device)
        def eval(logits):
            temperature = self.temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated
        
        self.train_param = [self.temperature]
        self.optimizer = torch.optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self
    
    
class CaGCN(nn.Module):
    def __init__(self, model, num_nodes, num_class, dropout_rate, device):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.cagcn = GCN(num_class, 1, 16, drop_rate=dropout_rate, num_layers=2)

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.graph_temperature_scale(logits, edge_index)
        return logits * F.softplus(temperature)

    def graph_temperature_scale(self, logits, edge_index):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagcn(logits, edge_index)
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits, data.edge_index)
            calibrated = logits * F.softplus(temperature)
            return calibrated

        self.train_param = self.cagcn.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self
    
    
from typing import Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree

def shortest_path_length(edge_index, mask, max_hop):
    """
    Return the shortest path length to the mask for every node
    """
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask).to(device)
    for hop in range(max_hop):
        current_hop = torch.nonzero(mask).to(device)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=device)
        for node in current_hop:
            node_mask = edge_index[0,:]==node
            nbrs = edge_index[1,node_mask]
            next_hop[nbrs] = True
        hop += 1
        # mask for the next hop shouldn't be seen before
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True
    return dist_to_train   

class CalibAttentionLayer(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            edge_index: Adj,
            num_nodes: int,
            train_mask: Tensor,
            dist_to_train: Tensor = None,
            heads: int = 8,
            negative_slope: float = 0.2,
            bias: float = 1,
            self_loops: bool = True,
            fill_value: Union[float, Tensor, str] = 'mean',
            bfs_depth=2,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.fill_value = fill_value
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        self.temp_lin = Linear(in_channels, heads,
                               bias=False, weight_initializer='glorot')

        # The learnable clustering coefficient for training node and their neighbors
        self.conf_coef = Parameter(torch.zeros([]))
        self.bias = Parameter(torch.ones(1) * bias)
        self.train_a = Parameter(torch.ones(1))
        self.dist1_a = Parameter(torch.ones(1))

        # Compute the distances to the nearest training node of each node
        dist_to_train = dist_to_train if dist_to_train is not None else shortest_path_length(edge_index, train_mask, bfs_depth)
        self.register_buffer('dist_to_train', dist_to_train)

        self.reset_parameters()
        if self_loops:
            # We only want to add self-loops for nodes that appear both as
            # source and target nodes:
            self.edge_index, _ = remove_self_loops(
                self.edge_index, None)
            self.edge_index, _ = add_self_loops(
                self.edge_index, None, fill_value=self.fill_value,
                num_nodes=num_nodes)

    def reset_parameters(self):
        self.temp_lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor]):
        N, H = self.num_nodes, self.heads

        # Individual Temperature
        normalized_x = x - torch.min(x, 1, keepdim=True)[0]
        normalized_x /= torch.max(x, 1, keepdim=True)[0] - \
                        torch.min(x, 1, keepdim=True)[0]

        # t_delta for individual nodes
        # x_sorted_scalar: [N, 1]
        x_sorted = torch.sort(normalized_x, -1)[0]
        temp = self.temp_lin(x_sorted)

        # Next, we assign spatial coefficient
        # a_cluster:[N]
        a_cluster = torch.ones(N, dtype=torch.float32, device=x[0].device)
        a_cluster[self.dist_to_train == 0] = self.train_a
        a_cluster[self.dist_to_train == 1] = self.dist1_a


        # For confidence smoothing
        conf = F.softmax(x, dim=1).amax(-1)
        deg = degree(self.edge_index[0, :], self.num_nodes)
        deg_inverse = 1 / deg
        deg_inverse[deg_inverse == float('inf')] = 0

        out = self.propagate(self.edge_index,
                             temp=temp.view(N, H) * a_cluster.unsqueeze(-1),
                             alpha=x / a_cluster.unsqueeze(-1),
                             conf=conf)
        sim, dconf = out[:, :-1], out[:, -1:]
        out = F.softplus(sim + self.conf_coef * dconf * deg_inverse.unsqueeze(-1))
        out = out.mean(dim=1) + self.bias 
        return out.unsqueeze(1)

    def message(
            self,
            temp_j: Tensor,
            alpha_j: Tensor,
            alpha_i: OptTensor,
            conf_i: Tensor,
            conf_j: Tensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int]) -> Tensor:
        """
        alpha_i, alpha_j: [E, H]
        temp_j: [E, H]
        """
        if alpha_i is None:
            print("alphai is none")
        alpha = (alpha_j * alpha_i).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        # Agreement smoothing + Confidence smoothing
        return torch.cat([
            (temp_j * alpha.unsqueeze(-1).expand_as(temp_j)),
            (conf_i - conf_j).unsqueeze(-1)], -1)

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}{self.out_channels}, heads={self.heads}')

class VS(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(num_classes))
        self.bias = nn.Parameter(torch.ones(num_classes))

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.vector_scale(logits)
        return logits * temperature + self.bias

    def vector_scale(self, logits):
        """
        Expand temperature to match the size of logits
        """
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))
        return temperature

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.vector_scale(logits)
            calibrated = logits * temperature + self.bias
            return calibrated

        self.train_param = [self.temperature]
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self
    
class GATS(nn.Module):
    def __init__(self, model, edge_index, num_nodes, train_mask, num_class, dist_to_train, heads, bias):
        super().__init__()
        self.model = model
        self.num_nodes = num_nodes
        self.cagat = CalibAttentionLayer(in_channels=num_class,
                                         out_channels=1,
                                         edge_index=edge_index,
                                         num_nodes=num_nodes,
                                         train_mask=train_mask,
                                         dist_to_train=dist_to_train,
                                         heads=heads,
                                         bias=bias)

    def forward(self, x, edge_index):
        logits = self.model(x, edge_index)
        temperature = self.graph_temperature_scale(logits)
        return logits / temperature

    def graph_temperature_scale(self, logits):
        """
        Perform graph temperature scaling on logits
        """
        temperature = self.cagat(logits).view(self.num_nodes, -1)
        return temperature.expand(self.num_nodes, logits.size(1))

    def fit(self, data, train_mask, test_mask, wdecay):
        self.to(device)
        def eval(logits):
            temperature = self.graph_temperature_scale(logits)
            calibrated = logits / temperature
            return calibrated

        self.train_param = self.cagat.parameters()
        self.optimizer = optim.Adam(self.train_param, lr=0.01, weight_decay=wdecay)
        fit_calibration(self, eval, data, train_mask, test_mask)
        return self
    
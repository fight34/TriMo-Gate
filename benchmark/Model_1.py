import math
import torch
import torch.nn as nn
import torch.nn.functional as F # noqa
from torch_scatter import scatter_sum


class PositionalEncoding(nn.Module):
    def __init__(self, length, d_model):
        super(PositionalEncoding, self).__init__()
        self.length = length
        self.d_model = d_model
        self.register_buffer('pos_table', self.get_sinusoid_encoding_table())

        assert self.d_model % 2 == 0, "d_model must be even for sinusoidal encoding"

    def get_sinusoid_encoding_table(self):
        position = torch.arange(0, self.length).float().unsqueeze(1)  # Shape: (n_position, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model))

        pos_table = torch.zeros(self.length, self.d_model)  # Shape: (length, d_model)，d_model must be an even number

        pos_table[:, 0::2] = torch.sin(position * div_term)  # sin for even positions
        pos_table[:, 1::2] = torch.cos(position * div_term)  # cos for odd positions

        return pos_table.unsqueeze(0)  # (1, length, d_model)

    def forward(self, x):
        return x * math.sqrt(self.d_model) + self.pos_table


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ff_layers = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(4 * d_model, d_model),
                                       )
        self.dropout = nn.Dropout(dropout)

    def forward(self, attn_input, key_padding_mask):

        # Multi-head attention
        norm_attn_intput = self.norm1(attn_input)
        attn_output, _ = self.attention(query=norm_attn_intput,
                                        key=norm_attn_intput,
                                        value=norm_attn_intput,
                                        key_padding_mask=key_padding_mask)
        output_1 = attn_input + self.dropout(attn_output)

        # feedforward
        ffn_output = self.ff_layers(self.norm2(output_1))
        output_2 = output_1 + self.dropout(ffn_output)

        return output_2


class SMILESCNN(nn.Module):
    def __init__(self, d_model, num_kernels, dropout=0.1):
        super(SMILESCNN, self).__init__()

        self.kernel_sizes = [1, 2, 3, 4, 5]
        self.out_channels = num_kernels

        # 创建多个不同 kernel size 的卷积层
        self.conv_layers = nn.ModuleList()
        for k in self.kernel_sizes:
            conv_block = nn.Sequential(
                nn.Conv1d(
                    in_channels=d_model,
                    out_channels=self.out_channels,  # Number of kernels
                    kernel_size=k,
                    stride=1
                ),
                nn.GroupNorm(1, self.out_channels),  # 修正：使用self.out_channels
                nn.GELU()
            )
            self.conv_layers.append(conv_block)

        total_channels = self.out_channels * len(self.kernel_sizes)

        # fully connected layer
        self.fc = nn.Sequential(nn.Linear(total_channels, total_channels // 2),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(total_channels // 2, d_model)
                                )

    def forward(self, inputs, key_padding_mask):
        # inputs: (B, L, D)
        # key_padding_mask: (B, L), True 表示 padding
        valid_mask = (~key_padding_mask).float()  # True/1 表示真实 token

        x = inputs.permute(0, 2, 1)  # adjust to the shape required for conv_1d: (batch_size, d_model, length)

        conv_outputs = []

        for conv, k in zip(self.conv_layers, self.kernel_sizes):
            conv_out = conv(x)  # (B, C, L-k+1)

            # 构造卷积窗口级 mask：窗口内 k 个 token 全部有效才算有效
            window_score = F.avg_pool1d(
                valid_mask.unsqueeze(1),
                kernel_size=k,
                stride=1
            ).squeeze(1)  # (B, L-k+1)

            window_valid = torch.eq(
                window_score,
                torch.ones_like(window_score)
            ).float()  # (B, L-k+1)

            window_valid = window_valid.unsqueeze(1)  # (B, 1, L-k+1)

            # masked mean pooling
            conv_out = conv_out * window_valid
            denom = window_valid.sum(dim=-1).clamp(min=1)  # (B, 1)
            pooled = conv_out.sum(dim=-1) / denom  # (B, C)

            conv_outputs.append(pooled)

        concatenated = torch.cat(conv_outputs, dim=1)
        outputs = self.fc(concatenated)

        return outputs


class SMILESModel(nn.Module):
    """SMILES data flow"""

    def __init__(self, vocab_size, length, d_model, num_heads, num_attn_layers, cnn_kernels):
        super(SMILESModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(length, d_model)
        self.attention_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads) for _ in range(num_attn_layers)
        ])
        self.cnn = SMILESCNN(d_model, cnn_kernels)

    def forward(self, input_index_vector):

        key_padding_mask = input_index_vector == 0

        embeddings = self.embedding(input_index_vector)

        encoder_input = self.pos_encoding(embeddings)

        # 位置编码后，padding 位置重新清零
        encoder_input = encoder_input.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        encoder_output = encoder_input
        for layer in self.attention_layers:
            encoder_output = layer(encoder_output, key_padding_mask=key_padding_mask)

            # 每层 Transformer 后，padding 位置重新清零
            encoder_output = encoder_output.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        smiles_emb = self.cnn(encoder_output, key_padding_mask)

        return smiles_emb


class DMPNN(nn.Module):
    """
    Directed Message Passing Neural Network
    """
    def __init__(self, d_node, d_edge, d_hidden, graph_num_layers, dropout=0.1):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_layers = graph_num_layers

        # 初始边特征
        self.edge_init = nn.Sequential(
            nn.Linear(2 * d_node + d_edge, d_hidden),
            nn.ReLU()
        )

        # 每层的 message update MLP
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(graph_num_layers)
        ])

        # 节点读出
        self.readout = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU()
        )

        self.atom_readout = nn.Linear(d_node, d_hidden)

    @torch.no_grad()
    def compute_rev_edge(self, edge_index, num_nodes):
        row, col = edge_index

        edge_key = row * num_nodes + col
        rev_key = col * num_nodes + row

        sorted_key, perm = edge_key.sort()
        rev_pos = torch.searchsorted(sorted_key, rev_key)

        rev_edge = perm[rev_pos]
        valid = (sorted_key[rev_pos] == rev_key)
        rev_edge[~valid] = -1

        return rev_edge

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        num_nodes = x.size(0)

        # Graph membership for batched molecular graphs
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
            num_graphs = int(batch.max().item()) + 1
        else:
            batch = x.new_zeros(num_nodes, dtype=torch.long)
            num_graphs = 1

        # Atom-feature fallback representation for every graph
        atom_sum = scatter_sum(x, batch, dim=0, dim_size=num_graphs)

        atom_count = scatter_sum(
            torch.ones(num_nodes, 1, device=x.device, dtype=x.dtype),
            batch,
            dim=0,
            dim_size=num_graphs
        ).clamp(min=1.0)

        atom_mean = atom_sum / atom_count
        atom_rep = self.atom_readout(atom_mean)  # (num_graphs, d_hidden)

        # If the whole batch has no edges, directly return atom fallback
        if edge_index.size(1) == 0:
            return atom_rep

        row, col = edge_index

        # Directed-edge initialization
        h0 = self.edge_init(torch.cat([x[row], x[col], edge_attr], dim=-1))
        h = h0

        # Reverse-edge lookup
        if not hasattr(data, "_rev_edge"):
            data._rev_edge = self.compute_rev_edge(edge_index, num_nodes)

        rev_edge = data._rev_edge
        valid_rev = rev_edge >= 0
        safe_rev = rev_edge.clamp(min=0)

        # Directed message passing
        for layer in range(self.num_layers):
            node_in_sum = scatter_sum(h, col, dim=0, dim_size=num_nodes)

            # Remove the immediately reverse edge contribution
            m = node_in_sum[row] - h[safe_rev] * valid_rev.unsqueeze(-1)

            m = self.message_mlps[layer](m)

            # Initial-state residual connection
            h = h0 + m

        # Edge-to-node and node-to-graph readout
        node_feat = scatter_sum(h, col, dim=0, dim_size=num_nodes)
        graph_rep_raw = scatter_sum(node_feat, batch, dim=0, dim_size=num_graphs)
        graph_rep = self.readout(graph_rep_raw)

        # Detect which graphs actually contain at least one edge
        edge_graph = batch[row]
        has_edge = torch.zeros(num_graphs, dtype=torch.bool, device=x.device)
        has_edge[edge_graph] = True

        # For edge-free graphs, replace readout with atom fallback
        graph_rep = torch.where(
            has_edge.unsqueeze(-1),
            graph_rep,
            atom_rep
        )

        return graph_rep


class MaskedInstanceNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mask):
        # x: (B, C, N)
        # mask: (B, N) 布尔类型，True 表示有效点
        mask = mask.unsqueeze(1).to(dtype=x.dtype)  # (B, 1, N)
        count = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1, 1)

        mean = (x * mask).sum(dim=-1, keepdim=True) / count  # (B, C, 1)
        # 计算方差时，需要先减去均值（只对有效点计算）
        diff = (x - mean) * mask
        var = (diff.pow(2) * mask).sum(dim=-1, keepdim=True) / count

        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        output = self.weight.unsqueeze(-1) * x_norm + self.bias.unsqueeze(-1)
        return output * mask


class TNet(nn.Module):
    """A small network for predicting 3×3 rotation matrices"""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, 1, bias=False)
        self.conv3 = nn.Conv1d(128, 1024, 1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, k * k)
        )

        self.bn1 = MaskedInstanceNorm1d(64)
        self.bn2 = MaskedInstanceNorm1d(128)
        self.bn3 = MaskedInstanceNorm1d(1024)

    def forward(self, x, lengths):
        """
        x: (batch, num_points, 3)
        return: (batch, 3, 3), rotation matrix
        """
        batch_size, num_points, _ = x.size()

        #  Generate mask: (batch, num_points), True for valid points
        mask = (torch.arange(num_points, device=x.device).unsqueeze(0) < lengths.unsqueeze(1))

        x = x.transpose(1, 2)          # (batch, k, num_points)
        x = F.relu(self.bn1(self.conv1(x), mask))
        x = F.relu(self.bn2(self.conv2(x), mask))
        x = F.relu(self.bn3(self.conv3(x), mask))

        #  Set features of invalid points to -inf so that max ignores them
        x = x.masked_fill(~mask.unsqueeze(1), float('-inf'))

        x = torch.max(x, 2, keepdim=True)[0]   # (batch, 1024, 1)
        x = x.view(batch_size, -1)             # (batch, 1024)
        x = self.fc(x)                       # (batch, k*k)

        # Reshape the output to (batch, 3, 3)
        matrix = x.view(batch_size, self.k, self.k)

        # Add the identity matrix as a bias to keep the initial transformation close to the identity
        identity = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        matrix = matrix + identity

        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        r0 = u @ vh  # (B, 3, 3) 严格正交，行列式 ≈ 1

        # 确保行列式为 +1
        det = torch.det(r0)  # (B,)
        # 当 det<0 时，对 Vh 的最后一行取反
        sign = torch.sign(det).view(-1, 1)  # (B, 1, 1)
        vh_fixed = vh.clone()
        vh_fixed[:, -1, :] *= sign  # 广播乘法
        r = u @ vh_fixed  # 纯旋转矩阵

        return r


class PointNet(nn.Module):
    """
    input: (batch_size, num_points, feature_dim)
    output: (batch_size, output_dim)
    """
    def __init__(self, d_point_input, d_point_cloud, d_point_output, dropout=0.1):
        super(PointNet, self).__init__()

        self.d_input = d_point_input
        self.d_point_cloud = d_point_cloud
        self.d_output = d_point_output

        self.tnet = TNet(k=3)  # T-Net transforms the coordinates

        self.embedding = nn.Sequential(
            nn.Linear(d_point_input, d_point_cloud // 2),
            nn.GELU(),
            nn.Linear(d_point_cloud // 2, d_point_cloud),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.output_projection = nn.Sequential(
            nn.Linear(3 * d_point_cloud, d_point_output),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, lengths):
        """
        input: (batch_size, num_points, input_dim)
        lengths: (batch_size,), The actual number of atoms in each molecule
        """
        batch_size, num_points, _ = x.shape  # num_points: the number after padding.

        coords = x[:, :, :3]  # (batch, num_points, 3)
        other_feats = x[:, :, 3:]  # (batch, num_points, d_point_input-3)

        rot_matrix = self.tnet(coords, lengths)  # (batch, 3, 3)
        rotated_coords = torch.matmul(coords, rot_matrix)  # (batch, num_points, 3)

        x = torch.cat([rotated_coords, other_feats], dim=-1)

        x = self.embedding(x)  # (batch_size, num_points, d_point_input) -> (batch_size, num_points, d_point_cloud)

        # Create masks to set the padding  to -∞ so that the max pool will ignore them.
        mask = (torch.arange(num_points, device=x.device)  # Create a sequence from 0 to num_points-1.
                .expand(batch_size, num_points)  # (num_points) -> (batch_size, num_points)
                < lengths.unsqueeze(1))  # (batch_size,) -> (batch_size, 1)
        mask = mask.unsqueeze(-1)  # (batch_size, num_points) -> (batch_size, num_points, 1)

        # max pool
        x_masked_inf = x.masked_fill(~mask, float('-inf'))  # ~mask:invert, i.e., True -> False and False -> to True
        max_pool, _ = torch.max(x_masked_inf, dim=1)  # (batch_size, d_point_output)

        # mean pool
        x_masked_zero = x.masked_fill(~mask, 0.0)
        sum_pool = torch.sum(x_masked_zero, dim=1)  # (batch_size, d_point_cloud)
        count = lengths.unsqueeze(-1).clamp(min=1)  # avoid division by zero
        mean_pool = sum_pool / count  # (batch_size, d_point_cloud)

        x = torch.cat([max_pool, mean_pool, sum_pool], dim=1)
        output = self.output_projection(x)  # (batch_size, d_point_output)

        return output


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion of multimodal fingerprints.
    """
    def __init__(self, d_smiles, d_graph, d_point_output, d_output, dropout=0.1):
        super().__init__()
        self.d_smiles = d_smiles
        self.d_graph = d_graph
        self.d_point = d_point_output

        total_dim = d_smiles + d_graph + d_point_output

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, d_output)
        )

    def forward(self, smiles_emb, graph_emb, point_cloud_emb):
        if smiles_emb is not None:
            batch_size, device = smiles_emb.size(0), smiles_emb.device
        elif graph_emb is not None:
            batch_size, device = graph_emb.size(0), graph_emb.device
        else:
            batch_size, device = point_cloud_emb.size(0), point_cloud_emb.device

        if smiles_emb is None:
            smiles_emb = torch.zeros(batch_size, self.d_smiles, device=device)
        if graph_emb is None:
            graph_emb = torch.zeros(batch_size, self.d_graph, device=device)
        if point_cloud_emb is None:
            point_cloud_emb = torch.zeros(batch_size, self.d_point, device=device)

        combined = torch.cat([smiles_emb, graph_emb, point_cloud_emb], dim=-1)
        return self.mlp(combined)


class WeightedSumFusion(nn.Module):
    """
    Fusion of multimodal fingerprints via linear summation
    """
    def __init__(self, d_smiles, d_graph, d_point_output, d_output, dropout=0.1):
        super(WeightedSumFusion, self).__init__()

        common_dim = d_smiles
        assert d_smiles == d_graph == d_point_output, "Dimensions of modalities must be equal."

        self.common_dim = common_dim

        self.norm_s = nn.LayerNorm(d_smiles)
        self.norm_g = nn.LayerNorm(d_graph)
        self.norm_p = nn.LayerNorm(d_point_output)

        self.smiles_fc = nn.Linear(d_smiles, common_dim)
        self.graph_fc = nn.Linear(d_graph, common_dim)
        self.point_fc = nn.Linear(d_point_output, common_dim)

        self.alpha_logits = nn.Parameter(torch.zeros(3))  # Define a learnable scaling factor for each modality.

        self.fc = nn.Sequential(
            nn.LayerNorm(common_dim),
            nn.Linear(common_dim, common_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(common_dim // 2, d_output)
        )

        self.weights = None

    def forward(self, smiles_emb, graph_emb, point_cloud_emb, return_weights=False):

        if smiles_emb is not None:
            batch_size = smiles_emb.size(0)
            device = smiles_emb.device
        elif graph_emb is not None:
            batch_size = graph_emb.size(0)
            device = graph_emb.device
        elif point_cloud_emb is not None:
            batch_size = point_cloud_emb.size(0)
            device = point_cloud_emb.device
        else:
            raise ValueError("Provide at least one modality.")

        zero = torch.zeros(batch_size, self.common_dim, device=device)

        modality_mask = torch.tensor([
            smiles_emb is not None,
            graph_emb is not None,
            point_cloud_emb is not None
        ], dtype=torch.bool, device=device)

        if smiles_emb is not None:
            trans_smiles = self.smiles_fc(self.norm_s(smiles_emb))
        else:
            trans_smiles = zero

        if graph_emb is not None:
            trans_graph = self.graph_fc(self.norm_g(graph_emb))
        else:
            trans_graph = zero

        if point_cloud_emb is not None:
            trans_point = self.point_fc(self.norm_p(point_cloud_emb))
        else:
            trans_point = zero

        logits = self.alpha_logits.masked_fill(~modality_mask, -1e9)
        weights = torch.softmax(logits, dim=0)

        self.weights = weights.detach()

        fused_emb = (
                weights[0] * trans_smiles +
                weights[1] * trans_graph +
                weights[2] * trans_point
        )

        output = self.fc(fused_emb)

        if return_weights:
            return output, weights

        return output


class GatedFusion(nn.Module):
    """
    Fusion of multimodal fingerprints via gating mechanism
    """
    def __init__(self, d_smiles, d_graph, d_point_output, d_output, dropout=0.1):
        super(GatedFusion, self).__init__()

        # Ensure all modality dimensions are equal
        assert d_smiles == d_graph == d_point_output, "Dimensions of modalities must be equal."
        self.d_model = d_smiles

        # LayerNorm for cross-attention and the global residual connection, and the dropout
        self.norm_S = nn.LayerNorm(self.d_model)
        self.norm_G = nn.LayerNorm(self.d_model)
        self.norm_P = nn.LayerNorm(self.d_model)
        self.norm_final = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

        # ----- linear projection layers for pairwise interaction -----
        self.proj_S = nn.Linear(self.d_model, self.d_model)
        self.proj_G = nn.Linear(self.d_model, self.d_model)
        self.proj_P = nn.Linear(self.d_model, self.d_model)

        # ----- Gating Network -----
        # Input: concatenation of three enhanced features -> (batch, 6*d_model)
        # Output: logits of three gating vectors -> (batch, 6*d_model)
        self.gate_mlp = nn.Sequential(
            nn.Linear(6 * self.d_model, 3 * self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(3 * self.d_model, 3 * self.d_model)
        )

        # ----- Final prediction head -----
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, d_output)
        )

        # Save gating weights for interpretability analysis (filled in during forward pass)
        self.gates = None  # tuple (g_S, g_G, g_P), each of shape: (batch, d_model)

    def forward(self, smiles_emb, graph_emb, point_cloud_emb, return_gates=False):
        """
            return_gates: Whether to return gated vectors
        Returns:
            output: (batch, d_output)
            gates (optional): tuple (g_S, g_G, g_P)
        """
        if smiles_emb is not None:
            batch_size, device = smiles_emb.size(0), smiles_emb.device
        elif graph_emb is not None:
            batch_size, device = graph_emb.size(0), graph_emb.device
        elif point_cloud_emb is not None:
            batch_size, device = point_cloud_emb.size(0), point_cloud_emb.device
        else:
            raise ValueError("Provide at least one feature of a modality")

        # Set missing modalities directly to zero and project each modality to interaction space
        if smiles_emb is not None:
            h_s = self.norm_S(smiles_emb)
            s_proj = self.proj_S(h_s)
        else:
            h_s = torch.zeros(batch_size, self.d_model, device=device)
            s_proj = torch.zeros_like(h_s)

        if graph_emb is not None:
            h_g = self.norm_G(graph_emb)
            g_proj = self.proj_G(h_g)
        else:
            h_g = torch.zeros(batch_size, self.d_model, device=device)
            g_proj = torch.zeros_like(h_g)

        if point_cloud_emb is not None:
            h_p = self.norm_P(point_cloud_emb)
            p_proj = self.proj_P(h_p)
        else:
            h_p = torch.zeros(batch_size, self.d_model, device=device)
            p_proj = torch.zeros_like(h_p)

        # Hadamard product interactions
        h_sg = s_proj * g_proj   # SMILES-Graph
        h_gp = g_proj * p_proj   # Graph-PointCloud
        h_ps = p_proj * s_proj   # PointCloud-SMILES

        # ---------- Feature‑Level Gate Generation ----------
        # Concatenate original + interactions
        concat_feats = torch.cat([h_s, h_g, h_p, h_sg, h_gp, h_ps], dim=-1)  # 6*d_model
        gate_logits = self.gate_mlp(concat_feats).view(batch_size, 3, self.d_model)  # (batch, 3, d_model)

        modality_mask = torch.tensor(
            [
                smiles_emb is not None,
                graph_emb is not None,
                point_cloud_emb is not None,
            ],
            dtype=torch.bool,
            device=device,
        ).view(1, 3, 1)

        gate_logits = gate_logits.masked_fill(
            ~modality_mask,
            torch.finfo(gate_logits.dtype).min,
        )

        gates = torch.softmax(gate_logits, dim=1)  # Softmax over the modality dimension
        gate_s, gate_g, gate_p = gates[:, 0, :], gates[:, 1, :], gates[:, 2, :]

        self.gates = (gate_s.detach(), gate_g.detach(), gate_p.detach())

        # ---------- Weighted Fusion ----------
        h_final = self.norm_final(gate_s * h_s + gate_g * h_g + gate_p * h_p)

        # Output prediction
        output = self.fc(h_final)

        if return_gates:
            return output, (gate_s, gate_g, gate_p)
        return output

    def get_modality_importance(self):
        """
        Call forward() first; returns a scalar overall modality importance for interpretability analysis
        """
        gate_s, gate_g, gate_p = self.gates
        imp_s = gate_s.mean(dim=-1)  # (batch,)
        imp_g = gate_g.mean(dim=-1)
        imp_p = gate_p.mean(dim=-1)
        return imp_s, imp_g, imp_p


class MultimodalModel(nn.Module):
    """
    Multimodal molecular representation learning mode
    """
    def __init__(self, vocab_size, smiles_length, d_smiles, smiles_num_heads, smiles_num_layers, cnn_kernels,  # SMILES
                 d_node, d_edge, d_hidden, graph_num_layers,  # Graph
                 d_point_input, d_point_cloud, d_point_output,  # Point cloud
                 d_output,
                 use_smiles, use_graph, use_point_cloud,
                 dropout=0.1):

        super(MultimodalModel, self).__init__()

        self.use_smiles = use_smiles
        self.use_graph = use_graph
        self.use_point_cloud = use_point_cloud
        self.num_modalities = sum([use_smiles, use_graph, use_point_cloud])
        assert self.num_modalities >= 1, "At least one modality must be activated."

        # SMILES Modality
        if self.use_smiles:
            self.smiles_model = SMILESModel(vocab_size, smiles_length, d_smiles, smiles_num_heads, smiles_num_layers,
                                            cnn_kernels)
        else:
            self.use_smiles = None

        # Graph Modality
        if self.use_graph:
            self.graph_model = DMPNN(d_node, d_edge, d_hidden, graph_num_layers)
        else:
            self.use_graph = None

        # Point cloud Modality
        if self.use_point_cloud:
            self.point_cloud_model = PointNet(d_point_input, d_point_cloud, d_point_output)
        else:
            self.use_point_cloud = None

        # Single-modal ablation: completely disable the fusion module and use a simple MLP head
        if self.num_modalities == 1:
            if self.use_smiles:
                input_dim = d_smiles
            elif self.use_graph:
                input_dim = d_hidden
            else:
                input_dim = d_point_output

            self.output_head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, d_output)
            )
            self.fusion_model = None
        else:
            # Fusion
            # Publication default: the interaction-augmented feature-level gate.
            self.fusion_model = GatedFusion(d_smiles, d_hidden, d_point_output, d_output)
            self.output_head = None

    def forward(self, smiles, graph, point_cloud, point_cloud_lengths, return_embeddings=False):

        smiles_emb, graph_emb, point_cloud_emb = None, None, None

        if self.use_smiles:
            smiles_emb = self.smiles_model(smiles)
        if self.use_graph:
            graph_emb = self.graph_model(graph)
        if self.use_point_cloud:
            point_cloud_emb = self.point_cloud_model(point_cloud, point_cloud_lengths)

        #  Select the forward path based on whether multimodality is present.
        if self.fusion_model is not None:
            output = self.fusion_model(smiles_emb, graph_emb, point_cloud_emb)
        else:
            if self.use_smiles:
                feat = smiles_emb
            elif self.use_graph:
                feat = graph_emb
            else:
                feat = point_cloud_emb
            output = self.output_head(feat)

        if return_embeddings:
            return output, (smiles_emb, graph_emb, point_cloud_emb)
        return output

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


class FusionBase(nn.Module):
    """
    Common normalization and prediction head for fusion ablations.

    Inputs are three (batch, d_model) tensors in SMILES/Graph/Point order.
    None disables a complete branch for the whole batch; per-sample missing
    modalities are not supported. All branch output dimensions must be equal.
    """
    def __init__(self, d_smiles, d_graph, d_point_output, d_output, dropout=0.1):
        super().__init__()
        dimensions = (d_smiles, d_graph, d_point_output, d_output)
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
               for dim in dimensions):
            raise ValueError("Feature and output dimensions must be positive integers.")
        if not d_smiles == d_graph == d_point_output:
            raise ValueError("All modality dimensions must be equal for fusion ablations.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        self.d_model = d_smiles
        self.norm_S = nn.LayerNorm(self.d_model)
        self.norm_G = nn.LayerNorm(self.d_model)
        self.norm_P = nn.LayerNorm(self.d_model)
        self.norm_final = nn.LayerNorm(self.d_model)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, max(1, self.d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, self.d_model // 2), d_output)
        )

        # Diagnostics only; never keep a training computation graph here.
        # The existing gate-export script checks .gates on every fusion type.
        self.gates = None
        self.weights = None

    @staticmethod
    def _validate_hidden_dim(hidden_dim):
        if (isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int)
                or hidden_dim <= 0):
            raise ValueError("hidden_dim must be a positive integer.")

    def _prepare_features(self, smiles_emb, graph_emb, point_cloud_emb):
        features = (smiles_emb, graph_emb, point_cloud_emb)
        names = ("SMILES", "Graph", "PointCloud")
        reference = next((feature for feature in features if feature is not None), None)
        if reference is None:
            raise ValueError("Provide at least one modality.")

        for name, feature in zip(names, features):
            if feature is None:
                continue
            if not isinstance(feature, torch.Tensor):
                raise TypeError(f"{name} features must be a torch.Tensor or None.")
            if feature.ndim != 2 or feature.size(-1) != self.d_model:
                raise ValueError(
                    f"{name} features must have shape (batch, {self.d_model}); "
                    f"received {tuple(feature.shape)}."
                )
            if not feature.is_floating_point():
                raise TypeError(f"{name} features must use a floating-point dtype.")
            if feature.size(0) != reference.size(0):
                raise ValueError("All modality tensors must have the same batch size.")
            if feature.device != reference.device or feature.dtype != reference.dtype:
                raise ValueError("All modality tensors must have the same device and dtype.")

        if reference.size(0) == 0:
            raise ValueError("Empty batches are not supported.")

        modality_mask = torch.tensor(
            [feature is not None for feature in features],
            dtype=torch.bool,
            device=reference.device
        )
        norms = (self.norm_S, self.norm_G, self.norm_P)
        normalized = [
            norm(feature) if feature is not None else reference.new_zeros(reference.shape)
            for norm, feature in zip(norms, features)
        ]
        return normalized, modality_mask

    def _predict(self, fused_emb):
        return self.fc(self.norm_final(fused_emb))


class ConcatFusion(FusionBase):
    """Concatenate normalized branches, then map 3*d_model back to d_model."""
    def __init__(self, d_smiles, d_graph, d_point_output, d_output,
                 dropout=0.1, hidden_dim=None):
        super().__init__(d_smiles, d_graph, d_point_output, d_output, dropout)
        hidden_dim = max(1, 3 * self.d_model // 2) if hidden_dim is None else hidden_dim
        self._validate_hidden_dim(hidden_dim)
        self.hidden_dim = hidden_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * self.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.d_model)
        )

    def forward(self, smiles_emb, graph_emb, point_cloud_emb):
        features, _ = self._prepare_features(smiles_emb, graph_emb, point_cloud_emb)
        fused_emb = self.fusion_mlp(torch.cat(features, dim=-1))
        return self._predict(fused_emb)


class MeanFusion(FusionBase):
    """Uniform average over the available, normalized modality features."""
    def forward(self, smiles_emb, graph_emb, point_cloud_emb, return_weights=False):
        features, modality_mask = self._prepare_features(
            smiles_emb, graph_emb, point_cloud_emb
        )
        stacked = torch.stack(features, dim=1)  # (batch, 3, d_model)
        weights = modality_mask.to(dtype=stacked.dtype)
        weights = weights / weights.sum()
        self.weights = weights.detach()
        fused_emb = (stacked * weights.view(1, 3, 1)).sum(dim=1)
        output = self._predict(fused_emb)
        if return_weights:
            return output, weights
        return output


class WeightedSumFusion(FusionBase):
    """
    Global learned scalar weights, shared by all molecules.

    Zero logits initialize this module to MeanFusion when their common
    normalization layers and prediction head have identical parameters.
    There are no private branch projections in this controlled variant.
    """
    def __init__(self, d_smiles, d_graph, d_point_output, d_output, dropout=0.1):
        super().__init__(d_smiles, d_graph, d_point_output, d_output, dropout)
        self.alpha_logits = nn.Parameter(torch.zeros(3))

    def forward(self, smiles_emb, graph_emb, point_cloud_emb, return_weights=False):
        features, modality_mask = self._prepare_features(
            smiles_emb, graph_emb, point_cloud_emb
        )
        stacked = torch.stack(features, dim=1)
        logits = self.alpha_logits.masked_fill(
            ~modality_mask, torch.finfo(self.alpha_logits.dtype).min
        )
        weights = torch.softmax(logits, dim=0)
        self.weights = weights.detach()
        fused_emb = (stacked * weights.view(1, 3, 1)).sum(dim=1)
        output = self._predict(fused_emb)
        if return_weights:
            return output, weights
        return output


class AttentionFusion(FusionBase):
    """
    Self-attention over three modality tokens, followed by masked mean pooling.

    Attention maps have shape (batch, num_heads, 3, 3). They are diagnostics,
    not the channel-wise routing weights returned by GatedFusion.
    """
    def __init__(self, d_smiles, d_graph, d_point_output, d_output,
                 dropout=0.1, num_heads=4):
        super().__init__(d_smiles, d_graph, d_point_output, d_output, dropout)
        if (isinstance(num_heads, bool) or not isinstance(num_heads, int)
                or num_heads <= 0 or self.d_model % num_heads != 0):
            raise ValueError("num_heads must be positive and divide d_model.")
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, smiles_emb, graph_emb, point_cloud_emb, return_weights=False):
        features, modality_mask = self._prepare_features(
            smiles_emb, graph_emb, point_cloud_emb
        )
        tokens = torch.stack(features, dim=1)  # (batch, 3, d_model)
        key_padding_mask = (~modality_mask).unsqueeze(0).expand(tokens.size(0), -1)
        attended, attention_weights = self.attention(
            query=tokens,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=return_weights,
            average_attn_weights=False
        )
        updated = tokens + self.attention_dropout(attended)
        # Missing queries can still attend to valid keys; exclude them as well.
        updated = updated.masked_fill(~modality_mask.view(1, 3, 1), 0.0)
        fused_emb = updated.sum(dim=1) / modality_mask.sum().to(dtype=updated.dtype)
        output = self._predict(fused_emb)

        self.attention_weights = None
        if return_weights:
            attention_weights = attention_weights.masked_fill(
                ~modality_mask.view(1, 1, 3, 1), 0.0
            )
            self.attention_weights = attention_weights.detach()
            return output, attention_weights
        return output


class GatedFusion(FusionBase):
    """
    Feature-level gating with optional explicit pairwise products.

    With pairwise interactions and the default hidden width, this preserves
    the original GatedFusion computation and state_dict parameter names.
    """
    def __init__(self, d_smiles, d_graph, d_point_output, d_output,
                 dropout=0.1, hidden_dim=None, use_pairwise_interactions=True):
        super().__init__(d_smiles, d_graph, d_point_output, d_output, dropout)
        if not isinstance(use_pairwise_interactions, bool):
            raise TypeError("use_pairwise_interactions must be bool.")
        self.use_pairwise_interactions = use_pairwise_interactions
        hidden_dim = 3 * self.d_model if hidden_dim is None else hidden_dim
        self._validate_hidden_dim(hidden_dim)
        self.hidden_dim = hidden_dim

        if self.use_pairwise_interactions:
            self.proj_S = nn.Linear(self.d_model, self.d_model)
            self.proj_G = nn.Linear(self.d_model, self.d_model)
            self.proj_P = nn.Linear(self.d_model, self.d_model)

        input_dim = (6 if self.use_pairwise_interactions else 3) * self.d_model
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3 * self.d_model)
        )

    def forward(self, smiles_emb, graph_emb, point_cloud_emb, return_gates=False):
        features, modality_mask = self._prepare_features(
            smiles_emb, graph_emb, point_cloud_emb
        )
        h_s, h_g, h_p = features
        gate_features = list(features)

        if self.use_pairwise_interactions:
            # Mask after projection so a missing branch contributes no bias.
            s_proj = self.proj_S(h_s).masked_fill(~modality_mask[0], 0.0)
            g_proj = self.proj_G(h_g).masked_fill(~modality_mask[1], 0.0)
            p_proj = self.proj_P(h_p).masked_fill(~modality_mask[2], 0.0)
            h_sg = s_proj * g_proj
            h_gp = g_proj * p_proj
            h_ps = p_proj * s_proj
            gate_features.extend([h_sg, h_gp, h_ps])

        concat_feats = torch.cat(gate_features, dim=-1)
        gate_logits = self.gate_mlp(concat_feats).view(h_s.size(0), 3, self.d_model)
        gate_logits = gate_logits.masked_fill(
            ~modality_mask.view(1, 3, 1), torch.finfo(gate_logits.dtype).min
        )
        gates = torch.softmax(gate_logits, dim=1)
        gate_s, gate_g, gate_p = gates.unbind(dim=1)
        self.gates = (gate_s.detach(), gate_g.detach(), gate_p.detach())

        fused_emb = gate_s * h_s + gate_g * h_g + gate_p * h_p
        output = self._predict(fused_emb)
        if return_gates:
            return output, (gate_s, gate_g, gate_p)
        return output

    def get_modality_importance(self):
        """Mean routing weight per modality and sample, not causal importance."""
        if self.gates is None:
            raise RuntimeError("Call forward() before reading modality importance.")
        return tuple(gate.mean(dim=-1) for gate in self.gates)


class GatedFusionWithoutInteractions(GatedFusion):
    """
    Feature-level gating using only the three normalized branch features.

    No pairwise projection layers are registered. The MLP can still learn
    cross-modal dependencies; only explicit pairwise products are removed.
    """
    def __init__(self, d_smiles, d_graph, d_point_output, d_output,
                 dropout=0.1, hidden_dim=None):
        super().__init__(
            d_smiles, d_graph, d_point_output, d_output,
            dropout=dropout,
            hidden_dim=hidden_dim,
            use_pairwise_interactions=False
        )


FUSION_CLASSES = {
    "concat": ConcatFusion,
    "mean": MeanFusion,
    "weighted": WeightedSumFusion,
    "attention": AttentionFusion,
    "gate_no_interaction": GatedFusionWithoutInteractions,
    "gate": GatedFusion
}


class MultimodalModel(nn.Module):
    """
    Multimodal molecular representation learning model.

    fusion_type: concat (Concat), mean (Avg), weighted (SW), attention (Attn),
                 gate_no_interaction (FLG), or gate (IA-FLG).
    Fusion is bypassed when only one modality is active. The three encoder
    output dimensions must be equal for the controlled fusion comparison.
    """
    def __init__(self, vocab_size, smiles_length, d_smiles, smiles_num_heads, smiles_num_layers, cnn_kernels,  # SMILES
                 d_node, d_edge, d_hidden, graph_num_layers,  # Graph
                 d_point_input, d_point_cloud, d_point_output,  # Point cloud
                 d_output,
                 use_smiles, use_graph, use_point_cloud,
                 dropout=0.1, fusion_type="gate"):

        super(MultimodalModel, self).__init__()

        self.use_smiles = use_smiles
        self.use_graph = use_graph
        self.use_point_cloud = use_point_cloud
        self.num_modalities = sum([use_smiles, use_graph, use_point_cloud])
        assert self.num_modalities >= 1, "At least one modality must be activated."

        if not isinstance(fusion_type, str) or fusion_type not in FUSION_CLASSES:
            raise ValueError(f"fusion_type must be one of {tuple(FUSION_CLASSES)}.")
        self.fusion_type = fusion_type if self.num_modalities > 1 else None

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
            # Select the fusion strategy; all variants share normalization and the head.
            self.fusion_model = FUSION_CLASSES[fusion_type](
                d_smiles, d_hidden, d_point_output, d_output, dropout=dropout
            )
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

import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, feature_vocab_sizes, embed_dim=8, dnn_hidden_units=[64, 32], dropout_rate=0.2):
        """
        DeepFM 完整模型
        :param feature_vocab_sizes: 字典，记录每个特征的词表大小 (如 {'I1': 10, 'C1': 1500})
        :param embed_dim: 连续向量的嵌入维度
        :param dnn_hidden_units: 列表，DNN 各隐藏层的神经元个数
        :param dropout_rate: 防止过拟合的 Dropout 比例
        """
        super(DeepFM, self).__init__()
        
        self.feature_names = list(feature_vocab_sizes.keys())
        self.num_features = len(self.feature_names)
        
        # ==========================================
        # 1. 构建 Embedding 层 (FM 与 DNN 共享这部分参数)
        # ==========================================
        # FM 的一阶权重 (相当于 embed_dim = 1)
        self.fm_1st_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size + 1, 1) 
            for feat, vocab_size in feature_vocab_sizes.items()
        })
        
        # FM的二阶交叉与 DNN 的共享输入 (embed_dim 通常设为 8, 16 等)
        self.fm_2nd_embeddings = nn.ModuleDict({
            feat: nn.Embedding(vocab_size + 1, embed_dim) 
            for feat, vocab_size in feature_vocab_sizes.items()
        })
        
        # 初始化 Embedding 权重
        for feat in self.feature_names:
            nn.init.xavier_uniform_(self.fm_1st_embeddings[feat].weight)
            nn.init.xavier_uniform_(self.fm_2nd_embeddings[feat].weight)

        # ==========================================
        # 2. 构建 DNN 深度网络层
        # ==========================================
        # DNN 的输入是将所有特征的二阶 Embedding 拼接在一起 (Flatten)
        # 输入维度 = 特征总数 * embed_dim
        input_dim = self.num_features * embed_dim
        
        dnn_layers = []
        for hidden_dim in dnn_hidden_units:
            dnn_layers.append(nn.Linear(input_dim, hidden_dim))
            # LayerNorm 在 batch_size=1 时也能稳定工作，避免 BatchNorm 的训练期报错。
            dnn_layers.append(nn.LayerNorm(hidden_dim))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim # 更新下一层的输入维度
            
        # DNN 的最后一层，输出一个标量打分
        dnn_layers.append(nn.Linear(input_dim, 1))
        
        # 使用 nn.Sequential 将网络层串联起来
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x_dict):
        """
        前向传播计算
        :param x_dict: 包含各特征输入索引的字典 {特征名: tensor(batch_size,)}
        """
        # --- 数据预处理：收集 Embedding ---
        fm_1st_embs = []
        fm_2nd_embs = []
        
        for feat in self.feature_names:
            # 提取一阶 Embedding，形状: (batch_size, 1) -> 扩展为 (batch_size, 1, 1)
            emb_1st = self.fm_1st_embeddings[feat](x_dict[feat]).unsqueeze(1)
            fm_1st_embs.append(emb_1st)
            
            # 提取二阶 Embedding，形状: (batch_size, embed_dim) -> 扩展为 (batch_size, 1, embed_dim)
            emb_2nd = self.fm_2nd_embeddings[feat](x_dict[feat]).unsqueeze(1)
            fm_2nd_embs.append(emb_2nd)
            
        # 将列表拼接成张量
        # 形状: (batch_size, num_features, 1)
        fm_1st_embs_tensor = torch.cat(fm_1st_embs, dim=1) 
        # 形状: (batch_size, num_features, embed_dim)
        fm_2nd_embs_tensor = torch.cat(fm_2nd_embs, dim=1) 

        # ==========================================
        # 模块一：FM 层计算
        # ==========================================
        # 1. 一阶计算: 对特征维度求和
        y_fm_1st = torch.sum(fm_1st_embs_tensor, dim=1) # (batch_size, 1)
        
        # 2. 二阶计算: O(n) 魔法化简 (和的平方 - 平方的和)
        sum_of_emb = torch.sum(fm_2nd_embs_tensor, dim=1) # (batch_size, embed_dim)
        square_of_sum = torch.pow(sum_of_emb, 2)
        
        square_of_emb = torch.pow(fm_2nd_embs_tensor, 2)
        sum_of_square = torch.sum(square_of_emb, dim=1)
        
        y_fm_2nd = 0.5 * (square_of_sum - sum_of_square)
        y_fm_2nd = torch.sum(y_fm_2nd, dim=1, keepdim=True) # (batch_size, 1)

        # ==========================================
        # 模块二：DNN 层计算
        # ==========================================
        # 将 Embedding 张量展平 (Flatten)，喂给 DNN
        # 从 (batch_size, num_features, embed_dim) 变为 (batch_size, num_features * embed_dim)
        dnn_input = fm_2nd_embs_tensor.view(fm_2nd_embs_tensor.size(0), -1)
        y_dnn = self.dnn(dnn_input) # (batch_size, 1)

        # ==========================================
        # 模块三：结果融合与输出
        # ==========================================
        # 将 FM 一阶、FM 二阶和 DNN 的输出相加
        y_total = y_fm_1st + y_fm_2nd + y_dnn
        
        # CTR 预估是一个二分类概率问题，必须经过 Sigmoid 将输出压缩到 0~1 之间
        out_prob = torch.sigmoid(y_total)
        
        # 展平输出，使其形状变为 (batch_size,)
        return out_prob.squeeze(-1)

# 快速验证
if __name__ == "__main__":
    # 假设我们有 3 个特征
    dummy_vocab_sizes = {'I1': 10, 'C1': 500, 'C2': 300}
    model = DeepFM(dummy_vocab_sizes, embed_dim=8, dnn_hidden_units=[64, 32])

    # 模拟输入 (Batch Size = 4)
    dummy_x = {
        'I1': torch.tensor([1, 5, 0, 9]),
        'C1': torch.tensor([100, 2, 499, 50]),
        'C2': torch.tensor([50, 100, 200, 299])
    }

    # 测试前向传播
    predictions = model(dummy_x)
    print(f"模型输出概率: {predictions.detach().numpy()}")
    print(f"输出形状: {predictions.shape}") # 预期输出形状应为 (4,)

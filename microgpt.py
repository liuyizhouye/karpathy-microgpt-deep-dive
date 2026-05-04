"""
用最小、最直接的方式，只靠 Python 标准库训练并运行一个 GPT。
这个文件就是完整算法本身：数据、分词、自动求导、模型、训练、生成都在这里。
真实工程里的深度学习框架主要是在把这些步骤做得更快、更省显存、更方便。

@karpathy
"""

import os       # 用来检查 input.txt 是否已经存在
import math     # 提供 log、exp 等数学函数
import random   # 用来打乱数据、初始化参数、按概率抽样
random.seed(42) # 固定随机种子：每次运行时结果更容易复现

# 准备训练数据 `docs`：这里是一组名字，每个名字是一条“文档”。
# 如果本地没有 input.txt，就自动下载 makemore 项目里的 names.txt。
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# 分词器 Tokenizer：把字符串转成整数序列，也能把整数转回字符。
# 这个极简 GPT 按“字符”分词：每个不同字符都有一个 token id。
uchars = sorted(set(''.join(docs))) # 数据中出现过的不同字符，对应 token id 0..n-1
BOS = len(uchars) # 特殊 token：BOS 表示一段文本的开始，也在这里复用为结束
vocab_size = len(uchars) + 1 # 词表大小：所有字符 token，再加 1 个 BOS token
print(f"vocab size: {vocab_size}")

# 自动求导 Autograd：
# Value 表示一个参与计算的标量。它会记住自己是怎么由别的 Value 算出来的，
# 所以最后可以从 loss 反向一路应用链式法则，得到每个参数的梯度。
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # 节省内存的小优化

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # 前向计算得到的标量数值
        self.grad = 0                   # loss 对这个数值的导数，反向传播时填入
        self._children = children       # 计算图里生成当前节点时用到的子节点
        self._local_grads = local_grads # 当前节点对每个子节点的局部导数

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        # 先把计算图排成“从输入到输出”的拓扑顺序。
        # 反向传播时再倒着走，才能保证每个节点收到完整梯度。
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1 # d(loss) / d(loss) = 1，从最终 loss 开始反传
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # 链式法则：loss 对 child 的梯度 += loss 对 v 的梯度 * v 对 child 的局部梯度
                child.grad += local_grad * v.grad

# 初始化模型参数：模型学到的“知识”最终都会存在这些数字里。
n_layer = 1     # Transformer 层数；越多越深，这里只用 1 层方便理解
n_embd = 16     # 向量宽度，也叫 embedding 维度；每个 token 会变成 16 个数
block_size = 16 # 最长上下文长度；注意这个名字数据集里最长名字是 15 个字符
n_head = 4      # 注意力头数量；多个头可以从不同角度看上下文
head_dim = n_embd // n_head # 每个注意力头分到的向量宽度
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
# state_dict 模仿 PyTorch 的命名：把每一组权重矩阵集中放在一个字典里。
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row] # 把所有参数摊平成一个列表，方便优化器逐个更新
print(f"num params: {len(params)}")

# 定义模型结构：输入当前 token 和位置，输出“下一个 token 是谁”的打分 logits。
# 结构大致参考 GPT-2，但为了极简做了简化：
# layernorm 换成 rmsnorm，不使用 bias，GeLU 激活换成 ReLU。
def linear(x, w):
    # 全连接层 / 矩阵乘法：把输入向量 x 乘以权重矩阵 w。
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    # 把任意实数打分转成概率分布，所有概率加起来等于 1。
    # 先减去最大值是为了数值稳定，避免 exp 太大。
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    # RMSNorm：把向量缩放到比较稳定的范围，让训练更容易。
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # token embedding 表示“这个字符是谁”，position embedding 表示“它在第几个位置”。
    tok_emb = state_dict['wte'][token_id] # 当前 token 的向量
    pos_emb = state_dict['wpe'][pos_id] # 当前位置的向量
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # 把内容信息和位置信息加在一起
    x = rmsnorm(x) # 这里不是多余的：残差连接会让反向传播也经过这一步

    for li in range(n_layer):
        # 1) 多头自注意力：
        # 当前 token 会“看见”前面已经处理过的 token，并决定该关注谁。
        x_residual = x
        x = rmsnorm(x)
        # q/k/v 是注意力机制里的三组向量：
        # q=query，表示“我想找什么”；k=key，表示“我有什么特征”；v=value，表示“真正要取走的信息”。
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        # keys/values 缓存了当前序列里之前所有位置的 k/v，生成下一个字符时就能复用上下文。
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            # 每个 head 只处理向量中的一小段，所以多个 head 可以并行关注不同模式。
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            # q 和每个 k 做点积，得到“当前 token 应该关注过去每个位置多少”的原始分数。
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            # 用注意力权重对 v 做加权平均，得到这个 head 的输出。
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        # 残差连接：把注意力输出加回原输入，帮助信息和梯度流动。
        x = [a + b for a, b in zip(x, x_residual)]
        # 2) MLP 前馈网络：
        # 对每个位置单独做非线性变换，让模型不只是“加权平均上下文”。
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    # lm_head 把最终向量映射回词表大小的 logits：
    # 每个 logit 是“下一个 token 是对应字符/BOS”的未归一化分数。
    logits = linear(x, state_dict['lm_head'])
    return logits

# Adam 优化器：根据梯度更新参数。
# m 和 v 是 Adam 的两个历史缓存，用来让更新更稳定。
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # 一阶动量缓存：类似“梯度的移动平均”
v = [0.0] * len(params) # 二阶动量缓存：类似“梯度平方的移动平均”

# 开始训练：反复让模型猜下一个字符，猜错就通过梯度下降修正参数。
num_steps = 1000 # 训练步数
for step in range(num_steps):

    # 每一步取一个名字，把它变成 token 序列，并在前后都加 BOS。
    # 例如 "emma" 会变成 [BOS, e, m, m, a, BOS]：
    # 第一个 BOS 表示开始，最后一个 BOS 表示结束。
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # 前向传播：从左到右喂 token，让模型预测下一个 token。
    # 每一次预测都会产生一个 loss，最后取平均。
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log() # 交叉熵 loss：正确答案概率越低，惩罚越大
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # 当前名字上所有位置的平均 loss，越低说明预测越准

    # 反向传播：计算每个参数对 loss 的影响，也就是梯度。
    loss.backward()

    # Adam 参数更新：按照梯度方向微调模型参数。
    lr_t = learning_rate * (1 - step / num_steps) # 线性降低学习率，训练越到后面步子越小
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # 偏差修正：训练初期 m/v 都从 0 开始，需要校正一下估计值。
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0 # 清空梯度，避免下一步训练时把旧梯度累加进去

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# 推理 / 生成：训练完后，让模型一个字符一个字符地“编”新名字。
temperature = 0.5 # 控制随机性/创造性；越低越保守，越高越发散
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS # 从“开始”token 起步
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        # 按概率抽下一个 token，而不是永远选概率最大的那个，这样能生成更多样的名字。
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break # 抽到 BOS 就当作名字结束
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")

"""纯 Jittor heavy PD-LTS 模型与 checkpoint 工具。

核心数据形状统一为 ``[batch, points, channels]``。模型先在 patch 内构建
KNN 图提取局部几何特征，再通过可逆单调流抑制噪声；三个 DenoiseFlow 串联
形成最终的 Stage A/B/C 去噪网络。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional, Union

import jittor as jt
from jittor import nn
import numpy as np


CODE_DIR = Path(__file__).resolve().parent
RELEASE_ROOT = CODE_DIR.parent
DEFAULT_CKPT = RELEASE_ROOT / "checkpoints" / "model_epoch_0009.pkl"


def resolve_path(path: Union[str, os.PathLike], *, must_exist: bool = False) -> Path:
    """统一解析命令行路径，避免脚本里到处拼相对目录。"""
    p = Path(path).expanduser()
    out = p if p.is_absolute() else RELEASE_ROOT / p
    if must_exist and not out.exists():
        raise FileNotFoundError(out)
    return out.resolve()


def set_cuda(device: str = "cuda") -> None:
    jt.flags.use_cuda = 0 if str(device).lower() == "cpu" else 1


def knn_idx(query, source, k):
    """KNN 索引，query/source: (B,N,C)，返回 (B,N,k)。"""
    if query.shape[-1] == 3:
        _, idx = jt.misc.knn(query, source, k)
    else:
        dist = ((query.unsqueeze(2) - source.unsqueeze(1)) ** 2).sum(-1)
        _, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    return idx


def knn_gather(feat, idx):
    """按 KNN 索引收集特征：feat (B,N,C), idx (B,M,k) -> (B,M,k,C)。"""
    bsz, _n, channels = feat.shape
    _, points, nnei = idx.shape
    idx_b = jt.arange(bsz).reshape(bsz, 1)
    flat = idx.reshape(bsz, points * nnei)
    out = feat[idx_b, flat]
    return out.reshape(bsz, points, nnei, channels)


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = jt.array([0.5])

    def execute(self, x):
        return (x * jt.sigmoid(x * nn.softplus(self.beta))) / 1.1


class PreConv(nn.Module):
    """把点特征与邻居差分编码成第一层局部几何特征。"""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel * 2, out_channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05),
        )

    def execute(self, f, idx):
        neigh = knn_gather(f, idx)
        f_tiled = f.unsqueeze(2).broadcast(neigh.shape)
        x = jt.concat([f_tiled, neigh - f_tiled], dim=-1)
        x = self.conv(x.permute(0, 3, 1, 2))
        return x.max(dim=-1).transpose(1, 2)


class EdgeConv(nn.Module):
    """在固定 KNN 图上聚合边特征，并按配置保留前层特征。"""

    def __init__(self, in_channel, hidden_channel, out_channel, concat=True):
        super().__init__()
        self.concat = concat
        if not concat:
            hidden_channel += 32
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channel * 2, hidden_channel, kernel_size=(1, 1)),
                    nn.BatchNorm2d(hidden_channel),
                    nn.LeakyReLU(0.05),
                ),
                nn.Sequential(
                    nn.Conv2d(hidden_channel, out_channel, kernel_size=(1, 1), bias=True),
                    nn.BatchNorm2d(out_channel),
                    nn.LeakyReLU(0.05),
                ),
            ]
        )

    def execute(self, f, idx):
        neigh = knn_gather(f, idx)
        f_tiled = f.unsqueeze(2).broadcast(neigh.shape)
        x = jt.concat([f_tiled, neigh - f_tiled], dim=-1).permute(0, 3, 1, 2)
        for conv in self.convs:
            x = conv(x)
        x = x.max(dim=-1).transpose(1, 2)
        return jt.concat([x, f], dim=-1) if self.concat else x


class FeatMergeUnit(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(in_channel, hidden_channel, kernel_size=1), nn.BatchNorm1d(hidden_channel), nn.ReLU()),
                nn.Sequential(nn.Conv1d(hidden_channel, out_channel, kernel_size=1), nn.BatchNorm1d(out_channel), nn.ReLU()),
            ]
        )

    def execute(self, x):
        x = x.transpose(1, 2)
        for conv in self.convs:
            x = conv(x)
        return x.transpose(1, 2)


class NoiseEdgeConv(nn.Module):
    """根据坐标邻域估计附加噪声通道，供可逆流联合建模。"""

    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        self.linear1 = nn.Linear(in_channel * 2, hidden_channel)
        self.linear2 = nn.Linear(hidden_channel, hidden_channel)
        self.linear3 = nn.Linear(in_channel, hidden_channel)
        self.linear4 = nn.Linear(hidden_channel, hidden_channel)
        self.linear5 = nn.Linear(hidden_channel, out_channel)

    def execute(self, f, idx):
        neigh = knn_gather(f, idx)
        f_tiled = f.unsqueeze(2).broadcast(neigh.shape)
        x = jt.concat([neigh, neigh - f_tiled], dim=-1)
        x = nn.relu(self.linear1(x))
        x = nn.relu(self.linear2(x)).max(dim=2)
        ff = nn.relu(self.linear4(nn.relu(self.linear3(f))))
        return self.linear5(x + ff)


def _l2_normalize(x, eps=1e-12):
    return x / jt.maximum(jt.sqrt((x * x).sum()), jt.array(eps))


class InducedNormLinearJT(nn.Module):
    """带谱范数约束的线性层，保证后续不动点求逆能够收敛。

    训练时用幂迭代更新 ``u/v/scale``；推理时直接使用权重中保存的状态，
    保证同一输入和参数得到稳定结果。
    """

    def __init__(self, in_features, out_features, bias=True, coeff=0.98, n_iterations=None, atol=1e-3, rtol=1e-3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        bound = math.sqrt(6.0 / ((1 + 5) * in_features))
        self.weight = (jt.rand((out_features, in_features)) * 2 - 1) * bound
        self.bias = jt.zeros((out_features,)) if bias else None
        self.u = _l2_normalize(jt.randn((out_features,))).stop_grad()
        self.v = _l2_normalize(jt.randn((in_features,))).stop_grad()
        self.scale = jt.zeros((1,)).stop_grad()

    def compute_weight(self, update=True, n_iterations=None):
        u, v, weight = self.u, self.v, self.weight
        if update:
            max_itrs = self.n_iterations if n_iterations is None else n_iterations
            max_itrs = 200 if max_itrs is None else max_itrs
            with jt.no_grad():
                u_, v_, w = u.detach(), v.detach(), weight.detach()
                for _ in range(max_itrs):
                    old_u, old_v = u_.clone(), v_.clone()
                    u_ = _l2_normalize(jt.matmul(w, v_))
                    v_ = _l2_normalize(jt.matmul(w.transpose(0, 1), u_))
                    err_u = jt.sqrt(((u_ - old_u) ** 2).sum()) / (u_.numel() ** 0.5)
                    err_v = jt.sqrt(((v_ - old_v) ** 2).sum()) / (v_.numel() ** 0.5)
                    if float(err_u.item()) < float(self.atol + self.rtol * u_.max().item()) and float(err_v.item()) < float(self.atol + self.rtol * v_.max().item()):
                        break
                self.u.assign(u_)
                self.v.assign(v_)
                u, v = u_, v_
        sigma = (u.detach() * jt.matmul(weight, v.detach())).sum()
        if update:
            self.scale.assign(sigma.detach().reshape(1))
        factor = jt.maximum(jt.array(1.0), sigma / self.coeff)
        return weight / factor

    def execute(self, x):
        weight = self.compute_weight(update=False)
        y = jt.matmul(x, weight.transpose(0, 1))
        return y + self.bias if self.bias is not None else y


class ActNorm(nn.Module):
    """可逆逐通道归一化。warm start 后 initialized=1，跳过批次初始化。"""

    def __init__(self, num_features):
        super().__init__()
        self.weight = jt.zeros((num_features,))
        self.bias = jt.zeros((num_features,))
        self.initialized = jt.array(0).stop_grad()

    def execute(self, x):
        if int(self.initialized.item()) == 0:
            with jt.no_grad():
                c = x.shape[2]
                x_t = x.detach().transpose(0, 2).reshape((c, -1))
                mean = x_t.mean(dim=1)
                var = jt.maximum(x_t.sqr().mean(dim=1) - mean.sqr(), jt.array(0.2))
                self.bias.assign(-mean)
                self.weight.assign(-0.5 * jt.log(var))
                self.initialized.assign(jt.array(1))
        return (x + self.bias.reshape(1, 1, -1)) * jt.exp(self.weight.reshape(1, 1, -1))

    def inverse(self, y):
        return y * jt.exp(-self.weight.reshape(1, 1, -1)) - self.bias.reshape(1, 1, -1)


class FCNet(nn.Module):
    def __init__(self, channel, idim, nhidden, preact):
        super().__init__()
        layers = []
        if preact:
            layers.append(Swish())
        last = channel
        for _ in range(nhidden):
            layers.append(InducedNormLinearJT(last, idim, bias=True))
            layers.append(Swish())
            last = idim
        layers.append(InducedNormLinearJT(last, channel, bias=True))
        self.nnet = nn.Sequential(*layers)

    def execute(self, x):
        return self.nnet(x)


_FP_ITERS = 12
_GRAD_ITERS = int(os.environ.get("FP_GRAD_ITERS", "1"))


def _find_root(gnet, y):
    """用固定 12 次不动点迭代求解可逆单调块的隐变量。"""
    n_detached = max(0, _FP_ITERS - _GRAD_ITERS)
    if n_detached > 0:
        with jt.no_grad():
            x = y - gnet(y)
            for _ in range(n_detached - 1):
                x = y - gnet(x)
        x = x.detach()
    else:
        x = y - gnet(y)
    for _ in range(_GRAD_ITERS):
        x = y - gnet(x)
    return x


class IMonotoneBlock(nn.Module):
    """可逆单调残差块，正向与逆向共用同一套不动点求解逻辑。"""

    def __init__(self, nnet):
        super().__init__()
        self.nnet = nnet

    def execute(self, x):
        s2 = math.sqrt(2)
        w = _find_root(lambda z: self.nnet(z), s2 * x)
        return s2 * w - x

    def inverse(self, y):
        s2 = math.sqrt(2)
        w = _find_root(lambda z: -self.nnet(z), s2 * y)
        return s2 * w - y


class FlowAssembly(nn.Module):
    def __init__(self, channel, idim, nhidden):
        super().__init__()
        self.chain = nn.ModuleList(
            [
                IMonotoneBlock(FCNet(channel, idim, nhidden, preact=False)),
                ActNorm(channel),
                IMonotoneBlock(FCNet(channel, idim, nhidden, preact=True)),
                ActNorm(channel),
            ]
        )

    def execute(self, x):
        for layer in self.chain:
            x = layer(x)
        return x

    def inverse(self, y):
        for i in range(len(self.chain) - 1, -1, -1):
            y = self.chain[i].inverse(y)
        return y


_FEAT_CFG = {
    "heavy": dict(
        in_channel_e=[16, 48, 80, 112, 144, 176, 96, 120, 144, 168],
        in_channel_a=[48, 80, 112, 144, 176, 96, 120, 144, 168, 96],
        out_channel=[32, 32, 32, 32, 32, 96, 24, 24, 24, 96],
        concat_off=(5, 9),
    )
}


class DenoiseFlow(nn.Module):
    """heavy 配置里的单个去噪阶段，三段堆叠后构成最终模型。

    每个阶段包含 10 层邻域特征注入和 10 个可逆 FlowAssembly。进入潜空间
    后将最后 ``cut_channel`` 个噪声维度置零，再通过逆变换恢复三维坐标。
    """

    def __init__(self, aug_channel=32, n_injector=10, cut_channel=16, nflow_module=10, num_neighbors=32, idim=64, nhidden=2):
        super().__init__()
        self.pc_channel = 3
        self.aug_channel = aug_channel
        self.n_injector = n_injector
        self.num_neighbors = num_neighbors
        self.cut_channel = cut_channel
        self.nflow_module = nflow_module
        channel = self.pc_channel + self.aug_channel
        cfg = _FEAT_CFG["heavy"]
        self.noise_params = NoiseEdgeConv(self.pc_channel, 32, self.aug_channel)
        self.PreConv = PreConv(self.pc_channel, 16)
        self.feat_Conv = nn.ModuleList()
        self.AdaptConv = nn.ModuleList()
        for i in range(self.n_injector):
            concat = i not in cfg["concat_off"]
            self.feat_Conv.append(EdgeConv(cfg["in_channel_e"][i], 64, cfg["out_channel"][i], concat=concat))
            self.AdaptConv.append(FeatMergeUnit(cfg["in_channel_a"][i], 64, channel))
        self.flow_assemblies = nn.ModuleList([FlowAssembly(channel, idim, nhidden) for _ in range(self.nflow_module)])

    def feat_extract(self, xyz):
        # patch 内先做 KNN 图，再逐层抽取局部几何特征。
        idx = knn_idx(xyz, xyz, self.num_neighbors)
        f = self.PreConv(xyz, idx)
        feats = []
        for i in range(self.n_injector):
            f = self.feat_Conv[i](f, idx)
            feats.append(self.AdaptConv[i](f))
        return feats

    def unit_coupling(self, xyz):
        idx = knn_idx(xyz, xyz, self.num_neighbors)
        return self.noise_params(xyz, idx)

    def f(self, x, inj_f):
        for i in range(self.nflow_module):
            if i < self.n_injector:
                x = x + inj_f[i]
            x = self.flow_assemblies[i](x)
        return x

    def g(self, z, inj_f):
        for i in range(self.nflow_module - 1, -1, -1):
            z = self.flow_assemblies[i].inverse(z)
            if i < self.n_injector:
                z = z - inj_f[i]
        return z

    def execute(self, x):
        inj_f = self.feat_extract(x)
        aug = self.unit_coupling(x)
        z = self.f(jt.concat([x, aug], dim=-1), inj_f)
        keep = self.pc_channel + self.aug_channel - self.cut_channel
        # 潜空间截断是模型真正执行“去噪”的位置，其余维度保留几何信息。
        z = jt.concat([z[:, :, :keep], jt.zeros_like(z[:, :, keep:])], dim=-1)
        return self.g(z, inj_f)[..., : self.pc_channel]

    def denoise(self, noisy_pc):
        return self.execute(noisy_pc)


class HeavyDenoiseFlow(nn.Module):
    """三段 heavy flow：训练监督每一段，推理只输出第三段坐标。"""

    def __init__(self):
        super().__init__()
        self.flows = nn.ModuleList([DenoiseFlow() for _ in range(3)])

    def forward_all(self, x):
        p, outs = x, []
        for flow in self.flows:
            p = flow(p)
            outs.append(p)
        return outs

    def execute(self, x):
        return self.forward_all(x)[-1]

    def denoise(self, noisy_pc):
        return self.execute(noisy_pc)


def _state_from_model(model: nn.Module) -> Dict[str, np.ndarray]:
    return {k: np.ascontiguousarray(v.numpy()) for k, v in model.state_dict().items()}


def _load_jittor_state(path: Path) -> Dict[str, np.ndarray]:
    """只通过 ``jt.load`` 读取模型权重。"""
    if path.suffix.lower() != ".pkl":
        raise ValueError(f"Jittor checkpoint must use .pkl, got: {path}")
    obj = jt.load(str(path))
    if not isinstance(obj, dict):
        raise ValueError(f"checkpoint must contain a dict: {path}")
    if "model_state" in obj:
        obj = obj["model_state"]
    out = {}
    for k, v in obj.items():
        out[k] = np.ascontiguousarray(v.numpy() if hasattr(v, "numpy") else np.asarray(v))
    return out


def load_state(model: nn.Module, ckpt: Union[str, os.PathLike], strict: bool = True) -> Dict[str, int]:
    """严格加载纯 Jittor state dict，逐项检查名称和张量形状。

    当前 Heavy 模型的正确加载结果必须是
    assigned=1998、missing=0、extra=0。
    """
    path = resolve_path(ckpt, must_exist=True)
    src = _load_jittor_state(path)
    dst = model.state_dict()
    assigned, missing, extra = 0, [], []
    for key, arr in src.items():
        if key not in dst:
            extra.append(key)
            continue
        target = dst[key]
        if tuple(target.shape) != tuple(arr.shape):
            if not strict and arr.size == target.numel():
                arr = arr.reshape(tuple(target.shape))
            else:
                raise ValueError(f"shape mismatch {key}: target {tuple(target.shape)} vs ckpt {arr.shape}")
        target.assign(jt.array(arr))
        assigned += 1
    for key in dst:
        if key not in src:
            missing.append(key)
    if strict and (missing or extra):
        raise ValueError(f"state mismatch: assigned={assigned} missing={len(missing)} extra={len(extra)}")
    if assigned == 0:
        raise ValueError(f"no checkpoint tensors assigned from {path}")
    return {"assigned": assigned, "missing": len(missing), "extra": len(extra)}


def save_state(model: nn.Module, path: Union[str, os.PathLike]) -> Path:
    """保存纯 Jittor state dict；最终文件只能通过 jt.load 读取。"""
    out = resolve_path(path)
    if out.suffix.lower() != ".pkl":
        out = out.with_suffix(".pkl")
    out.parent.mkdir(parents=True, exist_ok=True)
    jt.save(_state_from_model(model), str(out))
    return out


def update_lipschitz(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, InducedNormLinearJT):
            module.compute_weight(update=True)


def load_heavy_model(ckpt: Optional[Union[str, os.PathLike]] = None, device: str = "cuda", strict: bool = True) -> HeavyDenoiseFlow:
    set_cuda(device)
    model = HeavyDenoiseFlow()
    load_state(model, ckpt or DEFAULT_CKPT, strict=strict)
    model.eval()
    return model

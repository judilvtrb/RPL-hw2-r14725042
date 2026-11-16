"""Problem 3：Rapid Motor Adaptation (RMA) 的實作。

整體概念（對應 Kumar et al., RSS 2021；在手內操作上的應用見 Qi et al., "Hora", CoRL 2022）：

    第一階段（老師，teacher）
        真實物理參數 e（6 個數字）──[priv_encoder μ]──> z（8 維 latent）
        [短歷史觀察, 目標旋轉, z] ──[base policy π]──> 動作
        用一般的 PPO 訓練，μ 和 π 一起學。

    第二階段（學生，student）
        最近 H 步的觀察歷史 ──[adapt_encoder φ]──> ẑ
        訓練目標：ẑ ≈ z（MSE），π 和 μ 都凍結不動，只更新 φ。
        rollout 由學生自己執行（on-policy，跟 RMA 論文一樣）。

    評分時（只用學生）
        歷史 ──φ──> ẑ ──> π ──> 動作；完全不讀真實物理參數。

這個檔案提供兩個 class：

* ``RMAActor``：actor 網路。同一個 class 同時支援老師與學生兩種模式，
  所以兩者的 state_dict 欄位完全相同，老師的權重可以直接搬給學生。
  - ``algo.obs_groups.actor`` 包含 ``critic`` → 老師模式（讀真實物理參數，只能用來訓練）
  - 只有 ``actor`` → 學生模式（評分用，只讀 actor 觀察）
* ``RMAAdaptation``：第二階段用的「演算法」，繼承 rsl_rl 的 PPO，
  但把 PPO 的更新換成「讓 φ 去回歸老師的 z」。這樣第二階段也能直接用
  ``hw2 train`` 跑，checkpoint 格式和 log 都跟一般訓練一模一樣。

觀察向量的排列方式（已在 UniLab 1.5.2 上實測確認）：
    actor 群組 = [joint_pos 的 H 幀 | joint_target 的 H 幀 | object_pos 的 H 幀 | goal_rel_quat]
    每個 term 是一塊 (H, dim)，由舊到新排列，最後一幀是最新的。
    joint_target 每步 = 上一步 + 0.0417 × 動作，所以它的歷史就等於動作歷史。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict

# 6 個物理參數的「訓練範圍中心」與「半寬」，用來把它們正規化到大約 [-1, 1]。
# 順序：質量倍率、摩擦倍率、重心 x、重心 y、重心 z、馬達 gain 倍率。
_PRIV_CENTER = (1.0, 1.0, 0.0, 0.0, 0.0, 1.0)
_PRIV_HALF_RANGE = (0.5, 0.4, 0.01, 0.01, 0.01, 0.3)


class RMAActor(MLPModel):
    """RMA 的 actor：base policy π + 物理編碼器 μ（老師用）+ 歷史編碼器 φ（學生用）。

    rsl_rl 建立 actor 的方式跟 MLPModel 一樣，config 裡 ``algo.actor`` 底下的其他 key
    都會當成這裡的參數傳進來。

    Args:
        term_dims: actor 群組中「有歷史」的每個觀察 term 的維度，依照 config 的順序。
            預設 [16, 16, 3] = joint_pos、joint_target、object_pos。
        history_length: 上面每個 term 的歷史長度 H（config 裡的 history_length 必須一致）。
        short_length: base policy 只看最近幾幀（論文中的 x_t）；其餘的歷史給 φ 用。
        tail_dim: 接在歷史後面、沒有歷史的 term 總維度（goal_rel_quat 是 4）。
        latent_dim: z 的維度。
        priv_dim: 真實物理參數的數量（critic 群組最後 6 個數字）。
        priv_hidden_dims: μ 的隱藏層。
        adapt_channels: φ 的卷積通道數。
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
        term_dims: Sequence[int] = (16, 16, 3),
        history_length: int = 30,
        short_length: int = 3,
        tail_dim: int = 4,
        latent_dim: int = 8,
        priv_dim: int = 6,
        priv_hidden_dims: Sequence[int] = (64, 64),
        adapt_channels: int = 32,
    ) -> None:
        # 這些設定在 super().__init__() 裡就會用到（_get_obs_dim / _get_latent_dim），
        # 所以要先存起來。它們都是一般的 Python 值，不是 nn.Module，可以先設定。
        self.term_dims = [int(d) for d in term_dims]
        self.history_length = int(history_length)
        self.short_length = int(short_length)
        self.tail_dim = int(tail_dim)
        self.latent_dim = int(latent_dim)
        self.priv_dim = int(priv_dim)
        if not 1 <= self.short_length <= self.history_length:
            raise ValueError("short_length must be between 1 and history_length")

        # 父類別會建立：觀察正規化器（obs_normalizer）、輸出分佈（distribution）、
        # 以及 base policy π（self.mlp，輸入是 [短歷史, 目標, z]）。
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        # 老師模式：actor 的觀察群組裡有 critic（裡面最後 6 個數字是真實物理參數）。
        self.teacher_mode = "critic" in self.obs_groups
        # 第二階段會把正規化器凍結，讓 π 看到的輸入分佈跟老師訓練時一樣。
        self.freeze_normalizer = False

        # μ：真實物理參數 → z（tanh 讓 z 落在 [-1, 1]，學生比較好回歸）
        self.priv_encoder = nn.Sequential(
            MLP(self.priv_dim, self.latent_dim, list(priv_hidden_dims), activation), nn.Tanh()
        )
        self.register_buffer("priv_center", torch.tensor(_PRIV_CENTER[: self.priv_dim]))
        self.register_buffer("priv_half_range", torch.tensor(_PRIV_HALF_RANGE[: self.priv_dim]))

        # φ：歷史 → ẑ。結構仿照 RMA 論文：先對每一幀做線性嵌入，再用 1D 卷積看時間上的變化。
        feat = sum(self.term_dims)
        c = int(adapt_channels)
        self.adapt_conv = nn.Sequential(
            nn.Conv1d(feat, c, kernel_size=1),  # 每一幀各自嵌入
            nn.ELU(),
            nn.Conv1d(c, c, kernel_size=min(6, self.history_length), stride=3 if self.history_length >= 9 else 1),
            nn.ELU(),
            nn.Conv1d(c, c, kernel_size=min(3, max(1, self.history_length // 4)), stride=1),
            nn.ELU(),
        )
        with torch.no_grad():
            conv_out = self.adapt_conv(torch.zeros(1, feat, self.history_length)).numel()
        self.adapt_head = nn.Sequential(nn.Linear(conv_out, self.latent_dim), nn.Tanh())

    # ------------------------------------------------------------------
    # 父類別會呼叫的兩個函式：決定正規化器和 π 的輸入大小
    # ------------------------------------------------------------------
    def _get_obs_dim(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        groups = list(obs_groups[obs_set])
        actor_dim = int(obs["actor"].shape[-1])
        expected = self.history_length * sum(self.term_dims) + self.tail_dim
        if actor_dim != expected:
            raise ValueError(
                f"RMAActor: actor 觀察是 {actor_dim} 維，但 term_dims={self.term_dims}、"
                f"history_length={self.history_length}、tail_dim={self.tail_dim} 應該是 {expected} 維。"
                "請確認 config 裡 env.observations.policy.terms.*.history_length 和 algo.actor 的設定一致。"
            )
        # 正規化器只作用在 actor 群組上（老師的物理參數另外正規化）
        return groups, actor_dim

    def _get_latent_dim(self) -> int:
        return self.short_length * sum(self.term_dims) + self.tail_dim + self.latent_dim

    # ------------------------------------------------------------------
    # 把 actor 觀察拆成：短歷史、目標、完整歷史序列
    # ------------------------------------------------------------------
    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        h = self.history_length
        short_parts, seq_parts = [], []
        start = 0
        for d in self.term_dims:
            block = x[:, start : start + h * d].reshape(b, h, d)  # (B, H, d)，最後一幀最新
            short_parts.append(block[:, -self.short_length :, :].reshape(b, -1))
            seq_parts.append(block)
            start += h * d
        short = torch.cat(short_parts, dim=-1)
        tail = x[:, start:]
        seq = torch.cat(seq_parts, dim=-1)  # (B, H, sum(term_dims))
        return short, tail, seq

    def _normalized_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return self.obs_normalizer(obs["actor"])

    def priv_latent(self, obs: TensorDict) -> torch.Tensor:
        """老師的 z = μ(真實物理參數)。只有訓練時的 critic 群組裡才有這些數值。"""
        priv = obs["critic"][:, -self.priv_dim :]
        priv = (priv - self.priv_center) / self.priv_half_range
        return self.priv_encoder(priv)

    def adapt_latent(self, obs: TensorDict) -> torch.Tensor:
        """學生的 ẑ = φ(歷史)。只用 actor 觀察，評分時也能用。"""
        _, _, seq = self._split(self._normalized_actor_obs(obs))
        feats = self.adapt_conv(seq.transpose(1, 2))  # Conv1d 要 (B, 通道, 時間)
        return self.adapt_head(feats.flatten(1))

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
    ) -> torch.Tensor:
        """π 的輸入 = [短歷史, 目標, z 或 ẑ]。"""
        x = self._normalized_actor_obs(obs)
        short, tail, _ = self._split(x)
        z = self.priv_latent(obs) if self.teacher_mode else self.adapt_latent(obs)
        return torch.cat([short, tail, z], dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization and not self.freeze_normalizer:
            self.obs_normalizer.update(obs["actor"])  # type: ignore[operator]

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("RMAActor does not support TorchScript export")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("RMAActor does not support ONNX export")


def _resolve_checkpoint(path: str | Path) -> Path | None:
    """接受 model_*.pt 檔案，或是 run 資料夾（自動挑編號最大的 model）。"""
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        models = sorted(
            p.glob("model_*.pt"),
            key=lambda m: int(m.stem.split("_", 1)[1]) if m.stem.split("_", 1)[1].isdigit() else -1,
        )
        return models[-1] if models else None
    return None


class RMAAdaptation(PPO):
    """RMA 第二階段：用學生自己的 rollout，訓練 φ 去回歸老師的 z。

    設定方式（config 的 ``algo.algorithm``）::

        class_name: hw2_dexhand.student.rma:RMAAdaptation
        teacher_checkpoint: runs/adapt_teacher      # 老師的 run 資料夾或 model_*.pt
        adapt_learning_rate: 5.0e-4

    其餘 PPO 參數照樣接受（num_learning_epochs、num_mini_batches 會沿用）。
    評分程式也會建立這個 class，但只載入 actor 權重；那時候找不到老師檔案也沒關係。
    """

    def __init__(
        self,
        actor: RMAActor,
        critic: MLPModel,
        storage,
        teacher_checkpoint: str | None = None,
        adapt_learning_rate: float = 5.0e-4,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        if not isinstance(actor, RMAActor):
            raise TypeError("RMAAdaptation needs algo.actor.class_name = hw2_dexhand.student.rma.RMAActor")
        if actor.teacher_mode:
            raise ValueError(
                "第二階段的 actor 必須是學生模式：algo.obs_groups.actor 只能是 [actor]，不能包含 critic"
            )

        # 1) 載入老師的權重（π、μ、正規化器、動作分佈）；φ 保持隨機初始化
        self.teacher_loaded = False
        if teacher_checkpoint:
            ckpt = _resolve_checkpoint(teacher_checkpoint)
            if ckpt is None:
                print(
                    f"[RMAAdaptation] 警告：找不到老師 checkpoint '{teacher_checkpoint}'。"
                    "如果現在是在訓練第二階段，請確認路徑；如果是評分，可以忽略這個訊息。"
                )
            else:
                state = torch.load(ckpt, map_location=self.device, weights_only=False)["actor_state_dict"]
                state = {k: v for k, v in state.items() if not k.startswith(("adapt_conv.", "adapt_head."))}
                missing, unexpected = actor.load_state_dict(state, strict=False)
                bad = [k for k in missing if not k.startswith(("adapt_conv.", "adapt_head."))]
                if bad or unexpected:
                    raise RuntimeError(
                        f"老師 checkpoint {ckpt} 跟學生的網路結構對不上：缺少 {bad}，多出 {list(unexpected)}。"
                        "老師和學生的 history_length、hidden_dims、latent_dim 必須相同。"
                    )
                self.teacher_loaded = True
                print(f"[RMAAdaptation] 已載入老師權重：{ckpt}")

        # 2) 凍結 π、μ、正規化器；只有 φ 可以學
        actor.freeze_normalizer = True
        for p in actor.parameters():
            p.requires_grad_(False)
        self.adapt_params = list(actor.adapt_conv.parameters()) + list(actor.adapt_head.parameters())
        for p in self.adapt_params:
            p.requires_grad_(True)
        self.learning_rate = float(adapt_learning_rate)
        self.optimizer = torch.optim.Adam(self.adapt_params, lr=self.learning_rate)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """學生用「確定性」動作去跑環境（評分時也是確定性），收集它自己會遇到的狀態。"""
        super().act(obs)  # 填好 storage 需要的欄位（value、log prob 等，這裡用不到）
        actions = self.actor(obs).detach()  # 不加隨機雜訊
        self.transition.actions = actions
        return actions

    def update(self) -> dict[str, float]:
        """監督式學習：讓 φ(歷史) 接近 μ(真實物理參數)。"""
        obs_all = self.storage.observations.flatten(0, 1)
        n = int(obs_all.batch_size[0])
        mb = max(1, n // self.num_mini_batches)
        total, count = 0.0, 0
        target_std = 0.0
        for _ in range(self.num_learning_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.num_mini_batches):
                idx = perm[i * mb : (i + 1) * mb]
                batch = obs_all[idx]
                with torch.no_grad():
                    target = self.actor.priv_latent(batch)
                pred = self.actor.adapt_latent(batch)
                loss = F.mse_loss(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.adapt_params, self.max_grad_norm)
                self.optimizer.step()
                total += loss.item()
                count += 1
                target_std = target.std(dim=0).mean().item()
        self.storage.clear()
        return {"adapt_mse": total / max(count, 1), "teacher_latent_std": target_std}


__all__ = ["RMAActor", "RMAAdaptation"]

"""Advanced Challenge：模仿學習（Imitation Learning）。

專家（expert）= Problem 1 用 PPO 訓練好的 policy（例如 runs/so3_ball，eval 8.00 goals/ep）。
學生（student）= 一個全新、隨機初始化的同樣大小的網路，只靠「模仿專家的動作」來學，
完全不使用 reward。

支援兩種方法（config 的 ``algo.algorithm.mode``）：

* ``bc``（Behavior Cloning）：
    由專家操作環境，學生記下「在這個狀態，專家做了什麼動作」，用監督式學習去擬合。
    缺點：學生只看過專家會遇到的狀態，自己一犯錯就進到沒看過的狀態（covariate shift，
    Ross et al., 2011），不知道怎麼救回來。

* ``dagger``（Dataset Aggregation，Ross, Gordon & Bagnell, AISTATS 2011）：
    由「學生」操作環境（一開始以機率 β 讓專家代打，β 隨訓練線性降到 0），
    但每個狀態都請專家標註「這裡應該怎麼做」。學生因此會看到自己犯錯後的狀態，
    並學會專家在那些狀態下的修正動作。

實作方式：繼承 rsl_rl 的 PPO，把「收集資料」和「更新」換掉。這樣就能直接用
``hw2 train`` 跑，存出來的 checkpoint 跟一般 PPO 一模一樣，``hw2 eval`` 可以直接評分。

設定範例（config 的 ``algo.algorithm``）::

    class_name: hw2_dexhand.student.imitation:Imitation
    expert_checkpoint: runs/so3_ball   # 專家的 run 資料夾或 model_*.pt
    mode: dagger                       # 或 bc
    beta_decay_iterations: 100         # DAgger：β 從 1 線性降到 0 所需的 iterations
    imitation_learning_rate: 1.0e-3
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms import PPO
from tensordict import TensorDict

from hw2_dexhand.student.rma import _resolve_checkpoint


class Imitation(PPO):
    """用專家的動作當標籤，訓練學生 actor（BC 或 DAgger）。"""

    def __init__(
        self,
        actor,
        critic,
        storage,
        expert_checkpoint: str | None = None,
        mode: str = "dagger",
        beta_decay_iterations: int = 100,
        imitation_learning_rate: float = 1.0e-3,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        if mode not in ("bc", "dagger"):
            raise ValueError("mode must be 'bc' or 'dagger'")
        self.mode = mode
        self.beta_decay_iterations = max(1, int(beta_decay_iterations))
        self.iteration = 0

        # 專家：複製一份跟學生一樣的網路結構，再載入專家的權重（含觀察正規化的統計值）。
        self.expert = None
        if expert_checkpoint:
            ckpt = _resolve_checkpoint(expert_checkpoint)
            if ckpt is None:
                print(
                    f"[Imitation] 警告：找不到專家 checkpoint '{expert_checkpoint}'。"
                    "如果現在是在訓練，請確認路徑；如果是評分，可以忽略這個訊息。"
                )
            else:
                expert = copy.deepcopy(actor)
                state = torch.load(ckpt, map_location=self.device, weights_only=False)["actor_state_dict"]
                expert.load_state_dict(state, strict=True)
                expert.eval()
                for p in expert.parameters():
                    p.requires_grad_(False)
                self.expert = expert
                print(f"[Imitation] 已載入專家：{ckpt}（mode={self.mode}）")

        # 只訓練學生 actor；critic 不需要（模仿學習不用 reward）。
        self.learning_rate = float(imitation_learning_rate)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

    def _beta(self) -> float:
        """DAgger 中專家代打的機率：從 1 線性降到 0。BC 永遠是 1（全部由專家操作）。"""
        if self.mode == "bc":
            return 1.0
        return max(0.0, 1.0 - self.iteration / self.beta_decay_iterations)

    def act(self, obs: TensorDict) -> torch.Tensor:
        super().act(obs)  # 填好 storage 需要的欄位（這裡用不到）
        student_actions = self.actor(obs).detach()
        if self.expert is None:
            actions = student_actions
        else:
            expert_actions = self.expert(obs).detach()
            beta = self._beta()
            if beta >= 1.0:
                actions = expert_actions
            elif beta <= 0.0:
                actions = student_actions
            else:
                use_expert = torch.rand(student_actions.shape[0], 1, device=student_actions.device) < beta
                actions = torch.where(use_expert, expert_actions, student_actions)
        self.transition.actions = actions
        return actions

    def update(self) -> dict[str, float]:
        """監督式學習：學生的動作（mean）去擬合專家在同一個狀態下的動作。"""
        if self.expert is None:
            raise RuntimeError("Imitation 需要 expert_checkpoint 才能訓練")
        obs_all = self.storage.observations.flatten(0, 1)
        n = int(obs_all.batch_size[0])
        mb = max(1, n // self.num_mini_batches)
        total, count = 0.0, 0
        for _ in range(self.num_learning_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.num_mini_batches):
                batch = obs_all[perm[i * mb : (i + 1) * mb]]
                with torch.no_grad():
                    target = self.expert(batch)
                pred = self.actor(batch)
                loss = F.mse_loss(pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total += loss.item()
                count += 1
        self.storage.clear()
        beta = self._beta()
        self.iteration += 1
        return {"imitation_mse": total / max(count, 1), "dagger_beta": beta}


__all__ = ["Imitation"]

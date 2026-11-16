"""只在訓練時使用的觀察項目（Problem 2）。

⚠️ 這些 term 只能放在 critic 群組（老師和 critic 用），絕對不能放進 actor 群組。
評分時 critic 群組會被評分程式拿掉，而且評分程式會拒絕 actor 群組裡的非官方 term。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from hw2_dexhand.tasks.allegro_inhand.terms.object_view import active_object_ids


def object_onehot(env: Any, num_objects: int = 3) -> torch.Tensor:
    """目前手上拿的是哪個物體（one-hot：球 [1,0,0]、方塊 [0,1,0]、圓柱 [0,0,1]）。

    多物體場景中，第 i 個環境拿的物體固定是 i % 3（見 multi_object.py）。
    """
    ids = active_object_ids(env, int(num_objects))
    return F.one_hot(ids, int(num_objects)).to(torch.float32)


__all__ = ["object_onehot"]

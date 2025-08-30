# tools/shard_alexnet_lossless_v2.py
import json, hashlib
from collections import OrderedDict
from pathlib import Path
import torch

# ===== 配置 =====
REPO_ROOT  = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT
SRC        = MODELS_DIR / "AlexNet.pth"         # 原始 state_dict
OUT_DIR    = MODELS_DIR / "alexnet_shards"      # 输出目录
PREFIX     = "alexnet"                           # 前缀
SHARD_MB   = 70                                  # 每片目标上限（MB），建议 ≤ 80MB
VERIFY     = True                                # 写入 sha256 以便加载时核验

def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.element_size() * t.numel()

def _sha256_tensor(t: torch.Tensor) -> str:
    b = t.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(b).hexdigest()

def _split_tensor_along0(t: torch.Tensor, max_bytes: int):
    """若 t 太大，沿第0维切片，返回一个 list[tensor]（保证每块 <= max_bytes）"""
    if t.dim() == 0:
        # 标量直接返回
        return [t]
    elem = t.element_size()
    row_elems = t[0].numel() if t.dim() > 0 else 1
    row_bytes = row_elems * elem
    if row_bytes > max_bytes:
        # 单行就超限（极少见）：退化为按元素切，仍能保证
        step = max(1, max_bytes // elem)
        flat = t.view(-1)
        parts = []
        for i in range(0, flat.numel(), step):
            parts.append(flat[i:i+step].clone())
        # 恢复形状由加载器完成（我们会在元信息里记录原 shape）
        return parts
    # 正常情况：按“行”切
    step_rows = max(1, max_bytes // row_bytes)
    parts = []
    for i in range(0, t.shape[0], step_rows):
        parts.append(t[i:i+step_rows].clone())
    return parts

def main():
    assert SRC.exists(), f"源模型不存在：{SRC}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    state = torch.load(SRC, map_location="cpu")
    if not isinstance(state, OrderedDict):
        state = OrderedDict(state.items())

    shard_limit = SHARD_MB * 1024 * 1024

    shards_meta = []          # [{"file":..., "keys":[...]}]
    keys_meta   = {}          # k -> {"dtype":..., "shape":..., "parts":[{"key":..., "sha256":...}], "is_split":bool}
    cur = OrderedDict()
    cur_size = 0
    shard_id = 0

    def flush():
        nonlocal cur, cur_size, shard_id
        if not cur:
            return
        name = f"{PREFIX}-{shard_id:04d}.pth"
        torch.save(cur, OUT_DIR / name)
        shards_meta.append({"file": name, "keys": list(cur.keys())})
        shard_id += 1
        cur, cur_size = OrderedDict(), 0

    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        t = v.detach().cpu().contiguous()
        info = {"dtype": str(t.dtype), "shape": tuple(t.shape), "parts": [], "is_split": False}

        nbytes = _tensor_nbytes(t)
        if nbytes > shard_limit:
            # 大张量细分
            parts = _split_tensor_along0(t, shard_limit)
            info["is_split"] = True
            # 如果退化到扁平切片，记录原形状，加载时再 reshape
            info["flat_shape_restore"] = tuple(t.shape)

            for pi, p in enumerate(parts):
                pbytes = _tensor_nbytes(p)
                # 如果当前分片放不下，先落盘
                if cur and cur_size + pbytes > shard_limit:
                    flush()
                key_part = f"{k}|part{pi:04d}"
                cur[key_part] = p
                cur_size += pbytes
                part_meta = {"key": key_part}
                if VERIFY:
                    part_meta["sha256"] = _sha256_tensor(p)
                info["parts"].append(part_meta)

                # 如果刚好到上限，落盘
                if cur_size >= shard_limit:
                    flush()
        else:
            # 普通张量：尽量塞当前分片
            if cur and cur_size + nbytes > shard_limit:
                flush()
            cur[k] = t
            cur_size += nbytes
            one = {"key": k}
            if VERIFY:
                one["sha256"] = _sha256_tensor(t)
            info["parts"].append(one)

        keys_meta[k] = info

    flush()

    index = {
        "format": "sharded_state_dict_v2",
        "prefix": PREFIX,
        "shard_mb": SHARD_MB,
        "verify_hash": VERIFY,
        "num_shards": len(shards_meta),
        "keys_meta": keys_meta,
        "shards": shards_meta,
    }
    (OUT_DIR / f"{PREFIX}-index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[done] {len(shards_meta)} shards -> {OUT_DIR}")

if __name__ == "__main__":
    main()
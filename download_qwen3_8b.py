#!/usr/bin/env python3
"""下载 Qwen3-8B 模型到本地目录。"""

from __future__ import annotations

import argparse
from pathlib import Path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 Qwen3-8B 到本地")
    parser.add_argument(
        "--repo-id",
        default="Qwen/Qwen3-8B",
        help="Hugging Face 模型仓库 ID（默认: Qwen/Qwen3-8B）",
    )
    parser.add_argument(
        "--local-dir",
        default="models/Qwen3-8B",
        help="本地保存目录（默认: models/Qwen3-8B）",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="可选：下载指定分支/commit/tag",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="可选：Hugging Face 访问令牌（私有仓库或更高限额时使用）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="继续未完成下载（默认会自动复用缓存）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit("请先安装 huggingface_hub：pip install -U huggingface_hub") from exc

    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
        token=args.token,
        resume_download=args.resume,
    )
    print(f"✅ 下载完成，模型保存在: {path}")


if __name__ == "__main__":
    main()

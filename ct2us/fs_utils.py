"""小工具：带重试的原子写 + 跨进程文件锁。

用途：
  - 索引文件在 4 进程并发下必须"要么旧内容、要么新内容"，不能半写。
  - 多个 worker 可能同时想写同一个 CT 体积文件（`volumes/<case>_ct.nii.gz`），
    用锁串行化，避免半个文件被另一个进程读到。

只用标准库，跨平台。
"""
from __future__ import annotations

import contextlib
import errno
import json
import os
import time
from pathlib import Path

_LOCK_SUFFIX = ".lock"


@contextlib.contextmanager
def file_lock(path, timeout: float = 600.0, poll: float = 0.05):
    """对 `path` 加一个基于 create-exclusive 的跨进程锁。

    以 `<path>.lock` 的独立创建作为互斥；超时抛 TimeoutError。
    进程崩溃留下的陈旧锁（默认 30 分钟）会被清理。
    """
    lock_path = Path(str(path) + _LOCK_SUFFIX)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fd = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            # 清理陈旧锁
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > 1800:
                    os.unlink(lock_path)
                    continue
            except OSError:
                pass
            if time.time() - t0 > timeout:
                raise TimeoutError(f"获取锁超时: {lock_path}")
            time.sleep(poll)
    try:
        os.write(fd, str(os.getpid()).encode())
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(lock_path)
        except OSError:
            pass


def atomic_write_bytes(path, data: bytes) -> None:
    """原子写字节：先写 `<path>.tmp.<pid>`，再 os.replace（同盘原子）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path, obj, indent: int | None = 2) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent))


def write_index_jsonl(path, rows, key: str = "sid") -> int:
    """按 `key` 去重（后者覆盖前者，保留首次出现顺序）并原子重写 JSONL 索引。

    Returns: 写入的唯一行数。
    """
    order: list[str] = []
    by_key: dict[str, dict] = {}
    for r in rows:
        k = str(r.get(key, ""))
        if k not in by_key:
            order.append(k)
        by_key[k] = r          # 后面的覆盖前面的
    with file_lock(path):
        text = "".join(json.dumps(by_key[k], ensure_ascii=False) + "\n" for k in order)
        atomic_write_text(path, text)
    return len(order)


def read_index_jsonl(path, key: str = "sid") -> list[dict]:
    """读取索引并按 key 去重（保留首次出现顺序，值取最后一次出现）。"""
    path = Path(path)
    if not path.exists():
        return []
    order: list[str] = []
    by_key: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = str(r.get(key, ""))
            if k not in by_key:
                order.append(k)
            by_key[k] = r
    return [by_key[k] for k in order]

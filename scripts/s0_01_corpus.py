#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S0-1 语料装配

对应 docs/10_B词典构造方案.md §3.1。

只读四个原始文件的 `content` 字段，其余字段一律不读入内存。
产出 <out_dir>/corpus.jsonl，并写入 <out_dir>/audit.json 的 s0_01 段。

幂等：输入内容 sha1 与上次一致且 corpus.jsonl 存在时跳过；--force 强制重跑。

用法：
    python scripts/s0_01_corpus.py
    python scripts/s0_01_corpus.py --data_dir D:/ttcbm/data/raw --out_dir D:/ttcbm/data/lexicon
"""

import argparse
import bisect
import hashlib
import json
import os
import re
import statistics
import sys
import time
import zlib
from collections import Counter, defaultdict

STEP = "s0_01"

# (dataset, split, 相对路径)
SOURCES = (
    ("TOXICN", "train", os.path.join("TOXICN", "train.json")),
    ("TOXICN", "test", os.path.join("TOXICN", "test.json")),
    ("COLD", "train", os.path.join("COLD", "train.json")),
    ("COLD", "test", os.path.join("COLD", "test.json")),
)

ZERO_WIDTH_RE = re.compile("[\u200b-\u200d\u2060\ufeff]")
HSPACE_RE = re.compile("[ \t\u00a0\u3000\u2000-\u200a]+")
NEWLINE_RUN_RE = re.compile("\n{3,}")
CJK_RE = re.compile("[\u3400-\u4dbf\u4e00-\u9fff]")
WS_RE = re.compile(r"\s+")


def sha1_text(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def clean_text(s):
    """统一换行、去零宽字符、折叠水平空白、逐行 strip。

    保留换行：S0-2 要按换行分句。
    """
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = ZERO_WIDTH_RE.sub("", s)
    s = HSPACE_RE.sub(" ", s)
    s = "\n".join(line.strip() for line in s.split("\n"))
    s = NEWLINE_RUN_RE.sub("\n\n", s)
    return s.strip()


def cjk_ratio(s):
    """中文字符数 / 非空白字符数。"""
    body = WS_RE.sub("", s)
    if not body:
        return 0.0
    return len(CJK_RE.findall(body)) / len(body)


def read_json_list(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("%s 顶层不是 list，实际为 %s" % (path, type(data).__name__))
    return data


def shingle_hashes(text, k):
    """去掉空白后的字符 k-gram 的 32 位哈希集合。"""
    s = WS_RE.sub("", text)
    if not s:
        return frozenset()
    if len(s) <= k:
        return frozenset((zlib.crc32(s.encode("utf-8")),))
    return frozenset(
        zlib.crc32(s[i:i + k].encode("utf-8")) for i in range(len(s) - k + 1)
    )


class UnionFind(object):
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_dedup(records, args):
    """bottom-k 签名 + LSH banding + 精确 Jaccard 复核。返回需删除的下标集合。"""
    cand = [i for i, r in enumerate(records) if r["n_char"] >= args.near_dup_min_len]
    if not cand:
        return set(), {"candidates": 0, "pairs_compared": 0, "groups": 0, "dropped": 0}

    hs = dict((i, shingle_hashes(records[i]["text"], args.shingle_k)) for i in cand)

    buckets = defaultdict(list)
    for i in cand:
        sig = sorted(hs[i])[: args.sig_size]
        if not sig:
            continue
        # 按自身签名长度定 band 内行数，避免短文本只落在前 1-2 个 band
        rows = max(1, len(sig) // args.bands)
        for b in range(args.bands):
            part = tuple(sig[b * rows:(b + 1) * rows])
            if part:
                buckets[(b, part)].append(i)

    uf = UnionFind(len(records))
    n_pairs = 0
    for mem in buckets.values():
        if len(mem) < 2:
            continue
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                i, j = mem[a], mem[b]
                if uf.find(i) == uf.find(j):
                    continue
                A, B = hs[i], hs[j]
                if not A or not B:
                    continue
                n_pairs += 1
                inter = len(A & B)
                union = len(A) + len(B) - inter
                if union and inter / union >= args.jaccard:
                    uf.union(i, j)

    groups = defaultdict(list)
    for i in cand:
        groups[uf.find(i)].append(i)

    dropped = set()
    merged = 0
    for mem in groups.values():
        if len(mem) > 1:
            merged += 1
            keep = min(mem)  # 保留最早出现的一条
            for i in mem:
                if i != keep:
                    dropped.add(i)

    return dropped, {
        "candidates": len(cand),
        "pairs_compared": n_pairs,
        "groups": merged,
        "dropped": len(dropped),
    }


def assign_len_bins(records):
    vals = [r["n_char"] for r in records]
    if len(vals) >= 5 and len(set(vals)) >= 5:
        cuts = statistics.quantiles(vals, n=5, method="inclusive")
    else:
        lo = min(vals) if vals else 0
        cuts = [lo, lo + 1, lo + 2, lo + 3]
    for r in records:
        r["len_bin"] = bisect.bisect_right(cuts, r["n_char"])
    return [round(x, 3) for x in cuts], Counter(r["len_bin"] for r in records)


def load_audit(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_audit(path, audit):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def build(args):
    t0 = time.time()

    paths = [os.path.join(args.data_dir, rel) for _, _, rel in SOURCES]
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit("[ERROR] 缺少输入文件: %s" % p)

    combined = hashlib.sha1()
    input_files = {}
    for p in paths:
        with open(p, "rb") as f:
            b = f.read()
        h = hashlib.sha1(b).hexdigest()
        input_files[os.path.relpath(p, args.data_dir)] = h
        combined.update(h.encode("ascii"))
    input_sha1 = combined.hexdigest()

    out_path = os.path.join(args.out_dir, "corpus.jsonl")
    audit_path = os.path.join(args.out_dir, "audit.json")

    audit = load_audit(audit_path)
    prev = audit.get(STEP, {})
    if not args.force and prev.get("input_sha1") == input_sha1 and os.path.exists(out_path):
        print("[skip] 输入未变且 %s 已存在（--force 可强制重跑）" % out_path)
        return 0

    os.makedirs(args.out_dir, exist_ok=True)

    # 读入 + 清洗 + 剔除
    records = []
    per_file = {}
    for ds, split, rel in SOURCES:
        data = read_json_list(os.path.join(args.data_dir, rel))
        stat = Counter()
        stat["read"] = len(data)
        for pos, obj in enumerate(data):
            if not isinstance(obj, dict):
                stat["bad_record"] += 1
                continue
            raw = obj.get("content")
            if not isinstance(raw, str):
                stat["bad_record"] += 1
                continue
            text = clean_text(raw)
            if not text:
                stat["empty"] += 1
                continue
            if len(text) < args.min_char:
                stat["too_short"] += 1
                continue
            if cjk_ratio(text) < args.min_cjk:
                stat["low_cjk"] += 1
                continue
            stat["kept"] += 1
            records.append({
                "id": "%s-%s-%06d" % (ds, split, pos),
                "text": text,
                "dataset": ds,
                "split": split,
                "n_char": len(text),
                "sha1": sha1_text(text),
            })
        per_file["%s/%s" % (ds, split)] = dict(stat)

    n_after_filter = len(records)

    # 精确去重
    seen = set()
    uniq = []
    for r in records:
        if r["sha1"] in seen:
            continue
        seen.add(r["sha1"])
        uniq.append(r)
    n_exact = n_after_filter - len(uniq)
    records = uniq

    # 近似去重
    near_stats = {"candidates": 0, "pairs_compared": 0, "groups": 0, "dropped": 0}
    if args.near_dup == 1:
        dropped, near_stats = near_dedup(records, args)
        if dropped:
            records = [r for i, r in enumerate(records) if i not in dropped]

    # 长度分箱
    cuts, bin_counts = assign_len_bins(records)

    # 落盘
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "id": r["id"],
                "text": r["text"],
                "dataset": r["dataset"],
                "split": r["split"],
                "n_char": r["n_char"],
                "len_bin": r["len_bin"],
                "sha1": r["sha1"],
            }, ensure_ascii=False) + "\n")

    elapsed = time.time() - t0
    ds_counts = Counter(r["dataset"] for r in records)

    audit[STEP] = {
        "step": STEP,
        "params": {
            "data_dir": args.data_dir,
            "out_dir": args.out_dir,
            "min_char": args.min_char,
            "min_cjk": args.min_cjk,
            "near_dup": args.near_dup,
            "shingle_k": args.shingle_k,
            "sig_size": args.sig_size,
            "bands": args.bands,
            "jaccard": args.jaccard,
            "near_dup_min_len": args.near_dup_min_len,
        },
        "input_sha1": input_sha1,
        "input_files": input_files,
        "output_rows": len(records),
        "per_file": per_file,
        "stages": {
            "read_total": sum(v["read"] for v in per_file.values()),
            "after_filter": n_after_filter,
            "exact_dropped": n_exact,
            "near_dedup": near_stats,
            "final": len(records),
        },
        "len_quantile_cuts": cuts,
        "len_bin_counts": dict((str(k), v) for k, v in sorted(bin_counts.items())),
        "dataset_counts": dict(ds_counts),
        "elapsed_sec": round(elapsed, 3),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_audit(audit_path, audit)

    print("=" * 62)
    print("S0-1 语料装配完成   耗时 %.2fs" % elapsed)
    print("-" * 62)
    print("%-16s %8s %8s %8s %8s %8s" % ("文件", "读入", "坏记录", "空", "过短", "低中文"))
    for k, v in per_file.items():
        print("%-16s %8d %8d %8d %8d %8d"
              % (k, v.get("read", 0), v.get("bad_record", 0), v.get("empty", 0),
                 v.get("too_short", 0), v.get("low_cjk", 0)))
    print("-" * 62)
    print("过滤后 %d → 精确去重 -%d → 近似去重 -%d → 最终 %d"
          % (n_after_filter, n_exact, near_stats.get("dropped", 0), len(records)))
    print("数据集分布: %s" % dict(ds_counts))
    print("长度分位切点: %s" % cuts)
    print("长度箱分布: %s" % dict(sorted(bin_counts.items())))
    print("输出: %s" % out_path)
    print("审计: %s" % audit_path)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="S0-1 语料装配（只读 content 字段）")
    ap.add_argument("--data_dir", default=os.path.join("data", "raw"))
    ap.add_argument("--out_dir", default=os.path.join("data", "lexicon"))
    ap.add_argument("--min_char", type=int, default=2)
    ap.add_argument("--min_cjk", type=float, default=0.3)
    ap.add_argument("--near_dup", type=int, default=1, choices=(0, 1))
    ap.add_argument("--shingle_k", type=int, default=3)
    ap.add_argument("--sig_size", type=int, default=64)
    ap.add_argument("--bands", type=int, default=32)
    ap.add_argument("--jaccard", type=float, default=0.9)
    ap.add_argument("--near_dup_min_len", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return build(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

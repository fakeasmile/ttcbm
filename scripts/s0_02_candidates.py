#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S0-2 候选挖掘

对应 docs/10_B词典构造方案.md §3.2。

输入 <corpus_dir>/corpus.jsonl，输出 <out_dir>/candidates.csv，
并写入 <out_dir>/audit.json 的 s0_02 段。

用法：
    python scripts/s0_02_candidates.py
    python scripts/s0_02_candidates.py --corpus_dir data/lexicon --out_dir data/lexicon
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict

try:
    import jieba
    import jieba.posseg as pseg
    jieba.setLogLevel(logging.ERROR)
except ImportError:
    raise SystemExit("[ERROR] 需要 jieba（请在装有 jieba 的环境中运行）")

STEP = "s0_02"

# 单 token 允许的词性：名词类 + 动名兼类 + 动词 + 习用语/成语/简称/处所
SINGLE_POS_OK = frozenset(("n", "nr", "ns", "nt", "nz", "ng", "vn", "v", "l", "i", "j", "s", "b"))
# 多 token 禁止出现的词性**首字母**：代词/连词/介词/助词/副词/语气/叹词/量词
# 必须按首字母匹配：jieba 的助词是 uj/ul/ud/ug/uz 等子标签，精确匹配 "u" 会全部漏掉
MULTI_POS_BAD_HEAD = frozenset("rcpudyeoqxw")

INVALID_CHARS = re.compile("[^\u4e00-\u9fffA-Za-z0-9]")
CJK_ANY = re.compile("[\u4e00-\u9fff]")
TOKEN_PUNCT = re.compile(r"^[^\w]+$")
WS_RE = re.compile(r"\s+")

# 纯通用词，非指称物；长度超过 1 的词性过滤拦不住，用表兜
STOPWORDS = frozenset("""
东西 事情 问题 时候 地方 方面 情况 方式 样子 意思 感觉 想法 做法 结果 原因
一个 一些 一种 一样 这个 那个 这些 那些 这样 那样 什么 怎么 为什么 多少
我们 你们 他们 她们 它们 自己 大家 别人 人家 现在 以后 以前 今天 明天 昨天
因为 所以 但是 而且 如果 虽然 就是 不是 没有 可以 应该 需要 已经 还是
真的 确实 其实 当然 一定 可能 也许 大概 反正 到底 究竟 比如 例如 之类
""".split())


def read_corpus(path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def pos_ok(parts):
    """parts: [(word, flag), ...]"""
    if len(parts) == 1:
        return parts[0][1] in SINGLE_POS_OK
    for _, f in parts:
        if f and f[:1] in MULTI_POS_BAD_HEAD:
            return False
    return True


def split_sentences(tokens):
    """按标点 token 切句，避免跨句生成 n-gram。"""
    sents = []
    cur = []
    for w, f in tokens:
        if TOKEN_PUNCT.match(w):
            if cur:
                sents.append(cur)
                cur = []
        else:
            cur.append((w, f))
    if cur:
        sents.append(cur)
    return sents


def collect(docs, args):
    """统计每个候选的 tf / df / 首现词性 / 合法词性标记。"""
    info = {}
    token_tf = Counter()
    total_tokens = 0

    for d_idx, doc in enumerate(docs):
        text = doc.get("text", "")
        tokens = [(w, f) for w, f in pseg.cut(text)]
        for w, f in tokens:
            if not TOKEN_PUNCT.match(w):
                token_tf[w] += 1
                total_tokens += 1

        for sent in split_sentences(tokens):
            L = len(sent)
            for n in range(1, args.max_ngram + 1):
                for i in range(L - n + 1):
                    parts = sent[i:i + n]
                    term = "".join(w for w, _ in parts)
                    if not (args.min_len <= len(term) <= args.max_len):
                        continue
                    if INVALID_CHARS.search(term):
                        continue
                    if not CJK_ANY.search(term) and not term.isupper():
                        continue

                    ok = pos_ok(parts)
                    flags = ",".join(f for _, f in parts)
                    e = info.get(term)
                    if e is None:
                        info[term] = [1, 1, d_idx, n, flags,
                                      1 if ok else 0, 0 if ok else 1]
                    else:
                        e[0] += 1
                        if e[2] != d_idx:
                            e[1] += 1
                            e[2] = d_idx
                        if ok:
                            e[5] += 1
                        else:
                            e[6] += 1
                        if n < e[3]:
                            e[3] = n

    return info, token_tf, total_tokens


def pmi_score(term, tf, n_tok, token_tf, total_tokens):
    """log( p(term) / p(left)*p(right) )，多切分点取最小。单 token 候选返回 None（豁免）。"""
    if n_tok == 1:
        return None
    parts = [w for w, _ in pseg.cut(term)]
    if len(parts) < 2:
        return None
    pt = max(tf, 1) / total_tokens
    best = None
    for k in range(1, len(parts)):
        left = "".join(parts[:k])
        right = "".join(parts[k:])
        # 部件未作为独立 token 出现时按 1 计：0.5 不是合法计数，会把 PMI 系统性抬高
        pl = max(token_tf.get(left, 1), 1) / total_tokens
        pr = max(token_tf.get(right, 1), 1) / total_tokens
        v = math.log(pt / (pl * pr))
        if best is None or v < best:
            best = v
    return best


def entropy(counter):
    tot = float(sum(counter.values()))
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * math.log(c / tot) for c in counter.values() if c > 0)


def neighbor_entropy(docs, terms):
    """对给定候选集统计左右邻字熵。"""
    by_first = defaultdict(lambda: defaultdict(set))
    for t in terms:
        by_first[t[0]][len(t)].add(t)

    lc = defaultdict(Counter)
    rc = defaultdict(Counter)
    BOUND = "\x00"

    for doc in docs:
        s = WS_RE.sub("", doc.get("text", ""))
        n = len(s)
        for i in range(n):
            groups = by_first.get(s[i])
            if not groups:
                continue
            for L, bucket in groups.items():
                if i + L > n:
                    continue
                t = s[i:i + L]
                if t not in bucket:
                    continue
                lc[t][s[i - 1] if i > 0 else BOUND] += 1
                rc[t][s[i + L] if i + L < n else BOUND] += 1

    return ({t: entropy(c) for t, c in lc.items()},
            {t: entropy(c) for t, c in rc.items()})


def suppress_fragments(df_map, ratio):
    """A 是真子串 B 且 df_A/df_B <= ratio 时删 A。ratio 接近 1 才说明 A 被 B 覆盖。"""
    max_over = {}
    for b in sorted(df_map, key=len, reverse=True):
        db = df_map[b]
        L = len(b)
        for i in range(L):
            for j in range(i + 2, L + 1):
                if i == 0 and j == L:
                    continue
                s = b[i:j]
                if s in df_map and max_over.get(s, -1) < db:
                    max_over[s] = db
    return set(a for a, da in df_map.items()
               if a in max_over and da / float(max_over[a]) <= ratio)


def build(args):
    t0 = time.time()
    corpus_path = os.path.join(args.corpus_dir, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        raise SystemExit("[ERROR] 缺少 %s（先跑 S0-1）" % corpus_path)

    out_path = os.path.join(args.out_dir, "candidates.csv")
    audit_path = os.path.join(args.out_dir, "audit.json")

    audit = {}
    if os.path.exists(audit_path):
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                audit = json.load(f)
        except Exception:
            audit = {}

    with open(corpus_path, "rb") as f:
        corpus_sha1 = hashlib.sha1(f.read()).hexdigest()
    # 参数指纹也参与幂等判断，否则改了阈值仍会静默跳过
    param_sig = hashlib.sha1(json.dumps({
        "max_ngram": args.max_ngram, "min_len": args.min_len,
        "max_len": args.max_len, "min_df": args.min_df,
        "df_high": args.df_high, "single_char_df": args.single_char_df,
        "pmi_pct": args.pmi_pct, "pmi_abs": args.pmi_abs,
        "ent_pct": args.ent_pct, "ent_abs": args.ent_abs,
        "substr_ratio": args.substr_ratio,
    }, sort_keys=True).encode("utf-8")).hexdigest()
    prev = audit.get(STEP, {})
    if (not args.force and prev.get("input_sha1") == corpus_sha1
            and prev.get("param_sig") == param_sig and os.path.exists(out_path)):
        print("[skip] 输入与参数均未变且 %s 已存在（--force 可强制重跑）" % out_path)
        return 0

    docs = read_corpus(corpus_path)
    info, token_tf, total_tokens = collect(docs, args)
    funnel = {"docs": len(docs), "total_tokens": total_tokens, "unique_terms": len(info)}

    # 基础过滤：词性 + "的" 结尾 + 停用词 + 单字门槛
    cand = {}
    for term, e in info.items():
        tf, df, _, n, flags, ok_cnt, bad_cnt = e
        # 多数判定：偶发被标成名词不能放行（如"我"2866 次 r、个别次 n）
        if ok_cnt <= 0 or ok_cnt < bad_cnt:
            continue
        if term.endswith("的"):
            continue
        if term in STOPWORDS:
            continue
        if len(term) == 1 and df < args.single_char_df:
            continue
        if df < args.min_df:
            continue
        cand[term] = {"tf": tf, "df": df, "n_tok": n, "pos": flags,
                      "ok_n": ok_cnt, "bad_n": bad_cnt}
    funnel["after_basic"] = len(cand)

    # PMI
    for term, v in cand.items():
        v["pmi"] = pmi_score(term, v["tf"], v["n_tok"], token_tf, total_tokens)
    vals = [v["pmi"] for v in cand.values() if v["pmi"] is not None]
    pmi_cut = statistics.quantiles(vals, n=100)[args.pmi_pct - 1] if len(vals) >= 100 else None
    pmi_thr = max(pmi_cut or 0.0, args.pmi_abs)
    cand = dict((t, v) for t, v in cand.items()
                if v["pmi"] is None or v["pmi"] >= pmi_thr)
    funnel["after_pmi"] = len(cand)

    # 邻字熵
    ent_l, ent_r = neighbor_entropy(docs, set(cand))
    for term, v in cand.items():
        v["ent_l"] = ent_l.get(term, 0.0)
        v["ent_r"] = ent_r.get(term, 0.0)
    evals = [min(v["ent_l"], v["ent_r"]) for v in cand.values()]
    ent_cut = statistics.quantiles(evals, n=100)[args.ent_pct - 1] if len(evals) >= 100 else None
    ent_thr = max(ent_cut or 0.0, args.ent_abs)
    # 单字豁免分位门槛：单字不可能是更长词的碎片，只保留绝对下限
    cand = dict((t, v) for t, v in cand.items()
                if min(v["ent_l"], v["ent_r"]) >= (args.ent_abs if len(t) == 1 else ent_thr))
    funnel["after_entropy"] = len(cand)

    # 碎片抑制
    df_map = dict((t, v["df"]) for t, v in cand.items())
    drop = suppress_fragments(df_map, args.substr_ratio)
    for t in drop:
        cand.pop(t, None)
    funnel["after_fragment"] = len(cand)
    funnel["fragment_dropped"] = len(drop)

    os.makedirs(args.out_dir, exist_ok=True)
    rows = sorted(cand.items(), key=lambda kv: (-kv[1]["df"], kv[0]))
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "n_token", "df", "tf", "pmi", "ent_l", "ent_r",
                    "pos_seq", "ok_n", "bad_n", "low_freq"])
        for term, v in rows:
            w.writerow([
                term, v["n_tok"], v["df"], v["tf"],
                "" if v["pmi"] is None else round(v["pmi"], 4),
                round(v["ent_l"], 4), round(v["ent_r"], 4),
                v["pos"], v["ok_n"], v["bad_n"],
                1 if v["df"] < args.df_high else 0,
            ])

    elapsed = time.time() - t0
    audit[STEP] = {
        "step": STEP,
        "input_sha1": corpus_sha1,
        "param_sig": param_sig,
        "params": {
            "corpus_dir": args.corpus_dir,
            "out_dir": args.out_dir,
            "max_ngram": args.max_ngram,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "min_df": args.min_df,
            "df_high": args.df_high,
            "single_char_df": args.single_char_df,
            "pmi_pct": args.pmi_pct,
            "pmi_abs": args.pmi_abs,
            "ent_pct": args.ent_pct,
            "ent_abs": args.ent_abs,
            "substr_ratio": args.substr_ratio,
        },
        "funnel": funnel,
        "pmi_threshold": None if pmi_thr is None else round(pmi_thr, 4),
        "ent_threshold": None if ent_thr is None else round(ent_thr, 4),
        "output_rows": len(rows),
        "elapsed_sec": round(elapsed, 3),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print("=" * 62)
    print("S0-2 候选挖掘完成   耗时 %.2fs" % elapsed)
    print("-" * 62)
    print("文档 %d 条 / token %d / 唯一候选 %d" % (len(docs), total_tokens, len(info)))
    print("漏斗: 唯一 %d → 基础过滤 %d → PMI %d → 熵 %d → 碎片抑制 %d（删 %d）"
          % (len(info), funnel["after_basic"], funnel["after_pmi"],
             funnel["after_entropy"], funnel["after_fragment"],
             funnel["fragment_dropped"]))
    print("阈值: PMI ≥ %.3f / min(ent) ≥ %.3f" % (pmi_thr, ent_thr))
    print("单字候选 %d 个 / 低频档(df<%d) %d 个"
          % (sum(1 for t, _ in rows if len(t) == 1), args.df_high,
             sum(1 for _, v in rows if v["df"] < args.df_high)))
    print("输出: %s" % out_path)
    print("审计: %s" % audit_path)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="S0-2 候选挖掘")
    ap.add_argument("--corpus_dir", default=os.path.join("data", "lexicon"))
    ap.add_argument("--out_dir", default=os.path.join("data", "lexicon"))
    ap.add_argument("--max_ngram", type=int, default=3)
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=12)
    ap.add_argument("--min_df", type=int, default=5)
    ap.add_argument("--df_high", type=int, default=20)
    ap.add_argument("--single_char_df", type=int, default=50)
    ap.add_argument("--pmi_pct", type=int, default=70)
    ap.add_argument("--pmi_abs", type=float, default=2.0)
    ap.add_argument("--ent_pct", type=int, default=30)
    ap.add_argument("--ent_abs", type=float, default=0.5)
    ap.add_argument("--substr_ratio", type=float, default=1.1)
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

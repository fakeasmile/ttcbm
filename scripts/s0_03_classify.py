#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S0-3 候选归类

对应 docs/10_B词典构造方案.md §3.3。

输入 <out_dir>/candidates.csv
输出 <out_dir>/classified.csv、<out_dir>/s0_03_sample.csv
并合并写 <out_dir>/audit.json 的 s0_03 段。

用法：
    python scripts/s0_03_classify.py --model_path models/Qwen2.5-7B-Instruct
    python scripts/s0_03_classify.py --model_path models --backend hf
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

STEP = "s0_03"

# ---------------------------------------------------------------- 类目与判定树

CLASS_TABLE = """\
1  人物称谓：单数可指的人的标签（公知、圣母、小仙女、殖人、拳师）
2  群体范畴：集合名词、可加"们"（河南人、女性、凤凰男、小镇做题家）
3  亲属关系：关系项，亲属或伴侣（妈、祖宗、全家、孩子、老婆）
4  机构与行动者：有组织性/制度性身份的实体（官媒、资本、境外势力、水军、NGO）
5  观念与立场：被指称的观念、立场、主义（女权、政治正确、白左、普世价值）
6  身体与性：身体部位、生理特征、性（部位名、生理词、性行为词）
7  比拟物：被用作比拟的动物、物、污秽（狗、猪、蛆、韭菜、垃圾、屎）
8  疾病与缺陷：疾病、残障、能力缺陷（精神病、智障、残废、艾滋）
9  暴力与伤害手段：施于他人的伤害行为（打死、砍、强奸、人肉、开盒、举报、网暴）
10 死亡与自伤：指向自身或结果状态的死亡（死、自杀、跳楼、绝症、活该、下地狱）"""

DECISION_TREE = """\
Q1  是关系项（亲属/伴侣）                → 3
Q2  是有组织性/制度性身份的实体           → 4
Q3  是集合名词（可加"们"）且指人          → 2
Q4  是单数可指的人的标签                  → 1
Q5  是被指称的观念/立场/主义              → 5
Q6  是身体部位/生理特征/性                → 6
Q7  是疾病/残障/能力缺陷                  → 8
Q8  是被用作比拟的动物/物/污秽            → 7
Q9  是施于他人的伤害行为                  → 9
Q10 是指向自身或结果状态的死亡            → 10
否则                                      → 丢弃（两个字面/指称类都填 null）"""

EXAMPLE_LINE = ('{"term": "小仙女", "literal_class": 1, "referent_class": 1, '
                '"reason": "对年轻女性的贬称"}')

# 兜底规则用的种子词表（人工维护，可改）。只在 3 轮 LLM 都判不出时使用。
SEEDS = {
    1: ["公知", "圣母", "小仙女", "殖人", "拳师", "键盘侠", "杠精", "慕洋犬", "精日",
        "美分", "五毛", "舔狗", "直男癌", "普信男", "绿茶婊", "巨婴", "喷子", "海王",
        "捞女", "老赖", "渣男", "渣女", "白莲花", "走狗", "汉奸", "二鬼子"],
    2: ["河南人", "东北人", "黑人", "白人", "黄种人", "亚裔", "印度人", "上海人",
        "外地人", "农村人", "女性", "男性", "女人", "男人", "外国人", "中国人",
        "农民工", "富二代", "官二代", "凤凰男", "小镇做题家", "丁克", "剩女", "学生"],
    3: ["母亲", "父亲", "老婆", "老公", "媳妇", "丈夫", "妻子", "儿子", "女儿",
        "孩子", "全家", "祖宗", "家人", "亲戚", "兄弟", "姐妹", "婆婆", "丈母娘",
        "孙子", "孙女", "爹", "娘", "妈", "爸"],
    4: ["官媒", "资本", "境外势力", "水军", "政府", "媒体", "学校", "公司", "医院",
        "警察", "法院", "记者", "平台", "大厂", "国企", "外企", "军队", "警方",
        "教会", "联合国", "反对派"],
    5: ["女权", "政治正确", "白左", "普世价值", "田园女权", "女拳", "平权", "民族主义",
        "爱国主义", "自由主义", "保守主义", "环保主义", "消费主义", "拜金主义",
        "崇洋媚外", "双标", "道德绑架", "意识形态", "洗脑"],
    6: ["身体", "屁股", "胸", "腿", "皮肤", "身材", "身高", "体重", "处女", "性行为",
        "性取向", "性别", "生理", "月经", "子宫", "阴茎", "阴道", "乳房", "裸照",
        "卖淫", "嫖娼", "性"],
    7: ["畜生", "杂种", "禽兽", "蟑螂", "老鼠", "苍蝇", "蛆", "韭菜", "垃圾", "狗",
        "猪", "驴", "猴", "猩猩", "白菜", "工具人", "电池", "牛马", "社畜", "屎"],
    8: ["精神病", "智障", "脑残", "残废", "艾滋", "癌症", "抑郁", "疯子", "傻子",
        "瞎子", "聋子", "瘸子", "痴呆", "弱智", "低能", "畸形", "残疾", "传染病", "瘟疫"],
    9: ["打死", "砍", "杀", "强奸", "人肉", "开盒", "举报", "网暴", "霸凌", "殴打",
        "打人", "捅", "砸", "绑架", "拐卖", "性侵", "猥亵", "虐待", "毒打", "私刑", "封号"],
    10: ["死全家", "下地狱", "去死", "自杀", "跳楼", "绝症", "活该", "该死", "寻死",
         "轻生", "上吊", "割腕", "服毒", "猝死", "惨死", "死亡", "丧命", "火葬",
         "坟墓", "棺材", "死"],
}

SYSTEM_PROMPT = (
    "你是一名中文语料标注员。严格按给定判定树给候选词面归类，"
    "只输出 JSON 行，不解释、不评论。"
)


def build_user_prompt(terms):
    lines = []
    lines.append("下面是一棵按优先级排序的判定树，从上往下执行，先命中先定类：")
    lines.append(DECISION_TREE)
    lines.append("")
    lines.append("类目编号与内涵：")
    lines.append(CLASS_TABLE)
    lines.append("")
    lines.append("待判定候选（每行一个）：")
    for i, t in enumerate(terms, 1):
        lines.append("%d. %s" % (i, t))
    lines.append("")
    lines.append("要求：")
    lines.append("1. 每个候选先判“字面类”（这个词字面上像哪一类），再判“指称类”（在语料里实际指称的是哪一类）。"
                 "指称类必须按判定树顺序执行。")
    lines.append("2. 只判“指称的是什么”，不判褒贬、不判是否为侮辱。"
                 "同一个词可能字面是一类、指称是另一类（如“舔狗”字面是动物、指称是人）。")
    lines.append("3. 候选来自中文网络言论语料，可能是黑话、谐音、缩写或跨词碎片。"
                 "若它是碎片、不像任何一类的指称物，两个字面/指称类都填 null。")
    lines.append("4. 每个候选输出一行 JSON，字段固定为 term / literal_class / referent_class / reason。")
    lines.append("5. literal_class 与 referent_class 取 1-10 的整数或 null；reason 不超过 20 个字。")
    lines.append("6. term 必须与输入逐字一致，不要改写、不要带序号。")
    lines.append("7. 只输出 JSON 行，不要代码块，不要多余文字。示例：")
    lines.append(EXAMPLE_LINE)
    return "\n".join(lines)


# ---------------------------------------------------------------- 模型加载

def resolve_model_path(raw):
    p = Path(raw)
    if not p.exists():
        raise SystemExit("[ERROR] 模型路径不存在：%s（请用 --model_path 指定）" % p)
    if (p / "config.json").exists():
        return p
    subs = sorted(d for d in p.iterdir()
                  if d.is_dir() and (d / "config.json").exists())
    if len(subs) == 1:
        return subs[0]
    if not subs:
        raise SystemExit("[ERROR] %s 下没有含 config.json 的模型目录，"
                         "请用 --model_path 指定" % p)
    raise SystemExit("[ERROR] %s 下有多个模型，请用 --model_path 指定其一：\n  %s"
                     % (p, "\n  ".join(str(s) for s in subs)))


class ChatModel:
    """统一 generate(prompts) -> [str] 的接口。vLLM 优先，transformers 备用。"""

    def __init__(self, model_path, backend, gpu_mem, max_model_len):
        self.path = model_path
        self.backend = None
        if backend in ("auto", "vllm"):
            try:
                self._init_vllm(gpu_mem, max_model_len)
                self.backend = "vllm"
                return
            except ImportError:
                if backend == "vllm":
                    raise SystemExit("[ERROR] 指定 --backend vllm 但未安装 vllm")
        self._init_hf()
        self.backend = "hf"

    # -- vLLM
    def _init_vllm(self, gpu_mem, max_model_len):
        from vllm import LLM, SamplingParams  # noqa: F401
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path), trust_remote_code=True)
        kwargs = dict(model=str(self.path), trust_remote_code=True, dtype="auto",
                      gpu_memory_utilization=gpu_mem, enable_prefix_caching=True)
        if max_model_len and max_model_len > 0:
            kwargs["max_model_len"] = max_model_len
        print("[model] vLLM 加载 %s" % self.path)
        self.llm = LLM(**kwargs)
        self._vllm_sp = SamplingParams

    # -- transformers
    def _init_hf(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("[model] transformers 加载 %s" % self.path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.path), trust_remote_code=True,
            torch_dtype="auto", device_map="auto")
        self.model.eval()
        self.torch = torch

    def _format(self, user):
        tok = self.tokenizer
        combined = SYSTEM_PROMPT + "\n\n" + user
        if getattr(tok, "chat_template", None):
            # 有的中文 chat 模板不接受 system 角色，退化为单条 user
            for msgs in ([{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user}],
                         [{"role": "user", "content": combined}]):
                try:
                    return tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    continue
        return combined + "\n\n"

    def generate(self, user_prompts, max_new_tokens):
        texts = [self._format(u) for u in user_prompts]
        if self.backend == "vllm":
            sp = self._vllm_sp(temperature=0.0, max_tokens=max_new_tokens)
            outs = self.llm.generate(texts, sp)
            return [o.outputs[0].text if o.outputs else "" for o in outs]

        torch = self.torch
        res = []
        batch = 8
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 add_special_tokens=False)
            enc = {k: v.to(self.model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            res.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return res


# ---------------------------------------------------------------- 解析

LEAD_NUM = re.compile(r"^\s*\d+\s*[.、)．]\s*")
QUOTES = "\"'“”‘’「」"


def clean_term(s):
    if not isinstance(s, str):
        return ""
    s = s.strip().strip(QUOTES).strip()
    return LEAD_NUM.sub("", s).strip()


def as_class(v):
    """返回 (值, 是否合法)。合法值域：1..10 或 None。"""
    if v is None:
        return None, True
    if isinstance(v, bool):
        return None, False
    if isinstance(v, int):
        return (v, True) if 1 <= v <= 10 else (None, False)
    if isinstance(v, float) and v.is_integer():
        v = int(v)
        return (v, True) if 1 <= v <= 10 else (None, False)
    if isinstance(v, str):
        s = v.strip().strip(QUOTES).lower()
        if s in ("", "null", "none", "na", "n/a", "-"):
            return None, True
        if s.isdigit():
            n = int(s)
            return (n, True) if 1 <= n <= 10 else (None, False)
    return None, False


def extract_objects(text):
    """从模型输出里取出所有顶层 JSON 对象（容忍代码块、前后废话、嵌套）。"""
    if not text:
        return []
    body = text.replace("```json", "").replace("```", "")
    stripped = body.strip()
    if stripped.startswith("["):
        try:
            arr = json.loads(stripped)
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except Exception:
            pass
    out = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(body):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    try:
                        out.append(json.loads(body[start:i + 1]))
                    except Exception:
                        pass
                    start = -1
    return out


def parse_batch(text, expected):
    """返回 {term: (literal, referent, reason)}，只收合法且未重复的条目。"""
    got = {}
    for obj in extract_objects(text):
        term = clean_term(obj.get("term"))
        if term not in expected or term in got:
            continue
        # §3.3 步骤 3：字段缺失即 invalid，不能把"没给"当成"判为 null"
        if "literal_class" not in obj or "referent_class" not in obj:
            continue
        lit, ok1 = as_class(obj.get("literal_class"))
        ref, ok2 = as_class(obj.get("referent_class"))
        if not (ok1 and ok2):
            continue
        reason = obj.get("reason")
        reason = reason.strip()[:40] if isinstance(reason, str) else ""
        got[term] = (lit, ref, reason)
    return got


def make_record(term, lit, ref, reason, source):
    if ref is None:
        status = "drop" if source == "llm" else "unclassified"
        lit_out = lit
        transfer = 0
    else:
        status = "ok"
        lit_out = lit
        transfer = 1 if (lit is not None and lit != ref) else 0
    return {"term": term, "literal_class": lit_out, "class_id": ref,
            "transfer": transfer, "source": source, "status": status,
            "reason": reason}


def rule_classify(term):
    best = None  # (种子长度, -类号)
    for cid, seeds in SEEDS.items():
        for s in seeds:
            if s and s in term:
                key = (len(s), -cid)
                if best is None or key > best:
                    best = key
    return None if best is None else -best[1]


# ---------------------------------------------------------------- 主流程

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def read_candidates(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["term"], int(r["df"])))
    return rows


def load_partial(path, meta):
    """断点续跑：首行是 meta，不匹配则丢弃。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            head = f.readline()
            m = json.loads(head)
            if m != meta:
                return {}
            recs = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                recs[r["term"]] = r
            return recs
    except Exception:
        return {}


def append_partial(fp, recs):
    for r in recs:
        fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    fp.flush()
    os.fsync(fp.fileno())


def export_sample(records, args, path):
    rng = random.Random(args.seed)
    by_cls = defaultdict(list)
    for r in records.values():
        if r["status"] == "ok":
            by_cls[r["class_id"]].append(r)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "class_id", "source", "reason", "human_class"])
        for cid in range(1, 11):
            items = by_cls.get(cid, [])
            rule = [x for x in items if x["source"] == "rule"]
            llm = [x for x in items if x["source"] == "llm"]
            rng.shuffle(rule)
            rng.shuffle(llm)
            if rule and llm:
                # 两种来源都要抽到：规则判的占 1/3 名额（至多 10），其余给 LLM
                take_rule = min(len(rule), max(args.sample_per_class // 3, 5))
                sel = rule[:take_rule]
                sel += llm[:max(0, args.sample_per_class - len(sel))]
            else:
                sel = (rule + llm)[:args.sample_per_class]
            for r in sel:
                w.writerow([r["term"], cid, r["source"], r.get("reason", ""), ""])


def build(args):
    t0 = time.time()
    cand_path = os.path.join(args.out_dir, "candidates.csv")
    if not os.path.exists(cand_path):
        raise SystemExit("[ERROR] 缺少 %s（先跑 S0-2）" % cand_path)
    out_path = os.path.join(args.out_dir, "classified.csv")
    sample_path = os.path.join(args.out_dir, "s0_03_sample.csv")
    partial_path = os.path.join(args.out_dir, "s0_03_partial.jsonl")
    audit_path = os.path.join(args.out_dir, "audit.json")

    audit = {}
    if os.path.exists(audit_path):
        try:
            with open(audit_path, "r", encoding="utf-8") as f:
                audit = json.load(f)
        except Exception:
            audit = {}

    with open(cand_path, "rb") as f:
        input_sha1 = hashlib.sha1(f.read()).hexdigest()

    params = {"batch_size": args.batch_size, "retry_batch": args.retry_batch,
              "max_retries": args.max_retries, "max_new_tokens": args.max_new_tokens,
              "backend": args.backend, "model_path": str(args.model_path),
              "max_model_len": args.max_model_len}
    param_sig = hashlib.sha1(
        json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()

    prev = audit.get(STEP, {})
    if (not args.force and prev.get("input_sha1") == input_sha1
            and prev.get("param_sig") == param_sig
            and os.path.exists(out_path) and os.path.exists(sample_path)):
        print("[skip] 输入与参数均未变且输出已存在（--force 可强制重跑）")
        return 0

    rows = read_candidates(cand_path)
    terms = [t for t, _ in rows]
    df_map = dict(rows)
    print("[input] %d 个候选" % len(terms))

    meta = {"input_sha1": input_sha1, "param_sig": param_sig}
    if args.force:
        records = {}
    else:
        records = load_partial(partial_path, meta)
        if records:
            print("[resume] 从断点恢复 %d 条" % len(records))

    if any(t not in records for t in terms):
        model_path = resolve_model_path(args.model_path)
        model = ChatModel(model_path, args.backend, args.gpu_mem,
                          args.max_model_len)
        print("[model] backend=%s" % model.backend)
    else:
        model = None
        print("[resume] 全部条目已有记录，跳过模型加载")

    new_file = not os.path.exists(partial_path) or args.force or not records
    fp = open(partial_path, "w" if new_file else "a", encoding="utf-8")
    if new_file:
        fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
        fp.flush()

    rounds = []
    try:
        for rnd in range(args.max_retries + 1):
            pending = [t for t in terms if t not in records]
            if not pending:
                break
            size = args.batch_size if rnd == 0 else args.retry_batch
            batches = list(chunked(pending, size))
            rounds.append({"round": rnd, "pending": len(pending),
                           "batch_size": size, "batches": len(batches)})
            print("[round %d] 待判 %d 条 / %d 批（每批 %d）"
                  % (rnd, len(pending), len(batches), size))
            done_batches = 0
            for group in chunked(batches, args.chunk_batches):
                prompts = [build_user_prompt(b) for b in group]
                outs = model.generate(prompts, args.max_new_tokens)
                fresh = []
                for b, out in zip(group, outs):
                    got = parse_batch(out, set(b))
                    for term, (lit, ref, reason) in got.items():
                        if term in records:
                            continue
                        rec = make_record(term, lit, ref, reason, "llm")
                        records[term] = rec
                        fresh.append(rec)
                if fresh:
                    append_partial(fp, fresh)
                done_batches += len(group)
                print("  批 %d/%d | 已判 %d/%d"
                      % (done_batches, len(batches), len(records), len(terms)))
    finally:
        fp.close()

    # 规则兜底
    pending = [t for t in terms if t not in records]
    if pending:
        print("[rule] 规则兜底 %d 条" % len(pending))
        fresh = []
        for term in pending:
            cid = rule_classify(term)
            rec = make_record(term, None, cid, "", "rule")
            records[term] = rec
            fresh.append(rec)
        with open(partial_path, "a", encoding="utf-8") as f:
            append_partial(f, fresh)

    # 输出
    order = {t: i for i, t in enumerate(terms)}
    out_rows = sorted(records.values(),
                      key=lambda r: (r["class_id"] is None, r["class_id"] or 0,
                                     -df_map.get(r["term"], 0),
                                     order.get(r["term"], 0)))
    os.makedirs(args.out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "literal_class", "class_id", "transfer",
                    "source", "status"])
        for r in out_rows:
            w.writerow([r["term"],
                        "" if r["literal_class"] is None else r["literal_class"],
                        "" if r["class_id"] is None else r["class_id"],
                        r["transfer"], r["source"], r["status"]])

    export_sample(records, args, sample_path)

    cls_dist = {}
    for cid in range(1, 11):
        cls_dist[str(cid)] = sum(1 for r in records.values()
                                 if r["class_id"] == cid)
    status_dist = {}
    for s in ("ok", "drop", "unclassified"):
        status_dist[s] = sum(1 for r in records.values() if r["status"] == s)

    elapsed = time.time() - t0
    audit[STEP] = {
        "step": STEP,
        "input_sha1": input_sha1,
        "param_sig": param_sig,
        "params": params,
        "rounds": rounds,
        "resolved": {
            "llm": sum(1 for r in records.values() if r["source"] == "llm"),
            "rule": sum(1 for r in records.values() if r["source"] == "rule"),
        },
        "class_dist": cls_dist,
        "status_dist": status_dist,
        "transfer_count": sum(1 for r in records.values() if r["transfer"] == 1),
        "output_rows": len(out_rows),
        "elapsed_sec": round(elapsed, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    print("=" * 62)
    print("S0-3 归类完成   耗时 %.1fs" % elapsed)
    print("-" * 62)
    print("候选 %d → ok %d / drop %d / unclassified %d"
          % (len(terms), status_dist["ok"], status_dist["drop"],
             status_dist["unclassified"]))
    print("来源: LLM 判定 %d 条 / 规则兜底 %d 条"
          % (audit[STEP]["resolved"]["llm"], audit[STEP]["resolved"]["rule"]))
    print("各类: " + " ".join("%s:%d" % (k, v) for k, v in cls_dist.items()))
    print("转喻 transfer=1: %d" % audit[STEP]["transfer_count"])
    print("输出: %s" % out_path)
    print("抽检: %s（每类 %d 条，人工填 human_class）"
          % (sample_path, args.sample_per_class))
    print("审计: %s" % audit_path)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="S0-3 候选归类")
    ap.add_argument("--out_dir", default=os.path.join("data", "lexicon"))
    ap.add_argument("--model_path", default="models",
                    help="模型目录；可给含 config.json 的目录，或 models/ 下唯一子目录")
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=0,
                    help="0 表示不指定，用模型自身配置")
    ap.add_argument("--batch_size", type=int, default=40)
    ap.add_argument("--retry_batch", type=int, default=10)
    ap.add_argument("--chunk_batches", type=int, default=8,
                    help="每次建模调用并行提交的批数（每批写一次断点）")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=2560)
    ap.add_argument("--sample_per_class", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
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

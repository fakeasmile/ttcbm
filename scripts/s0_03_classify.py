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
import unicodedata
from collections import defaultdict
from pathlib import Path

STEP = "s0_03"

# ---------------------------------------------------------------- 进度条


class _PlainBar(object):
    """tqdm 缺失时的退化进度条：每 10% 往 stderr 打一行，不改变迭代结果。"""

    def __init__(self, iterable=None, total=None, desc="", **kwargs):
        self.iterable = iterable
        self.desc = desc or ""
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        self.total = total
        self.n = 0
        self.postfix = ""
        self._next = 0.0
        self._t0 = time.time()

    def __iter__(self):
        if self.iterable is None:
            return
        for item in self.iterable:
            yield item
            self.update(1)

    def update(self, k=1):
        self.n += k
        self._report()

    def _report(self, force=False):
        if not self.total:
            return
        pct = float(self.n) / self.total
        if not force and pct < self._next:
            return
        while self._next <= pct:
            self._next += 0.1
        tail = (" " + self.postfix) if self.postfix else ""
        sys.stderr.write("[%s] %d/%d (%.0f%%) %.0fs%s\n"
                         % (self.desc, self.n, self.total,
                            pct * 100, time.time() - self._t0, tail))
        sys.stderr.flush()

    def set_postfix_str(self, s="", refresh=True):
        self.postfix = s

    def set_description(self, s=None, refresh=True):
        if s is not None:
            self.desc = s

    def close(self):
        self._report(force=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None


def progress(iterable=None, total=None, desc="", **kwargs):
    """统一进度条入口：有 tqdm 用它，没有退化为每 10% 一行。"""
    if _tqdm is None:
        return _PlainBar(iterable, total=total, desc=desc)
    return _tqdm(iterable, total=total, desc=desc,
                 dynamic_ncols=True, leave=False, **kwargs)


# ---------------------------------------------------------------- 类目与判定树

ORDERED_RULES = """\
从 Q1 开始逐条问，第一个回答"是"的就停下，取它后面的类号。

Q1  是亲属、伴侣等关系项？                                          → 3 类    妈、孩子、老婆、祖宗
Q2  是有组织形式或制度身份的主体？（机构、组织、公司、媒体、资本、势力） → 4 类    官媒、资本、水军、世界女权组织
Q3  是指一个群体（不是一个具体个人）？                              → 2 类    女性、男性、黑人、白人、河南人、女、男
Q4  是称呼某一类个人的标签（单数可指）？                            → 1 类    公知、小仙女、圣母、毒贩、司机
Q5  是被指称的观念、立场、主义、制度？（抽象，不指具体人或组织）    → 5 类    女权、政治正确、白左、法西斯
Q6  是身体部位、生理特征、性？                                      → 6 类    屁股、月经、性行为
Q7  是疾病、残障、能力缺陷？（不指人、不指行为）                    → 8 类    智障、精神病、艾滋、残废
Q8  是被用来指人或贬人的动物、物、污秽？（普通器物、普通动物名不算） → 7 类    狗、猪、蛆、韭菜、畜生
Q9  是施于他人的伤害或致死行为？                                    → 9 类    打死、弄死、强奸、开盒、网暴
Q10 是自身死亡、自伤？（程度夸张不算）                              → 10 类   死、自杀、跳楼、身亡"""

NONE_RULES = """\
判 none 只适用于下面五种；逐条走完 Q1–Q10 都不成立时，也可以判 none：
A 虚词、代词、副词、连词、助词、语气词：是、有、说、会、要、能、就、都
B 被切断的残片（以 的/了/得 起头或结尾、单独不成词）：的人、说的是、的对
C 地名、国名、省名本身：北京、日本、江苏（"地名+人"如河南人走 Q3）
D 纯抽象过程、时间、数量：认为、出来、以后、一半、过于、接近
E 程度夸张与口头固定搭配（字面不指实体）：笑死、累死、狗屁、狗头

不许把具体实体判成 none。下面这些必须归类，判 none 就是错：
大汉、黄人、牛、工厂、族裔、男性、白人、学生、老人、智障、猪。
由常见语素组成的普通名词不是残片：黑人、印度人、亚裔 都正常归类。"""

SCHEMA_LINES = """\
每个候选输出一行 JSON，四个字段都要有：term / q / literal_class / reason；referent_class 可选。
- q 必填，只能是 Q1……Q10 或 none
- literal_class 必填：这个词「字面上」是什么（1-10）。字面与指称相同时也照实填同一个数
  （黑人字面就是人，填 2）；q=none 时填 null
- referent_class 可省略；写了必须与 q 对应：Q1→3 Q2→4 Q3→2 Q4→1 Q5→5 Q6→6 Q7→8 Q8→7 Q9→9 Q10→10，
  不一致整条作废
- reason 必须写出这个词本身并说明它指什么，≤20 字，不许把类目名抄进来
  错误："小姐姐指单数可指的人的标签"   正确："小姐姐指年轻女性"
- term 与输入逐字一致，不加序号；只输出 JSON 行，不要解释、不要代码块

写完自查三点：q 与 referent_class 对应吗；literal_class 填了吗；reason 里有这个词吗。"""

EXAMPLES = """\
{"term": "妈", "q": "Q1", "literal_class": 3, "reason": "妈是亲属关系"}
{"term": "黑人", "q": "Q3", "literal_class": 2, "reason": "黑人指一个种族群体"}
{"term": "男性", "q": "Q3", "literal_class": 2, "reason": "男性指男性群体"}
{"term": "圣母", "q": "Q4", "literal_class": 1, "reason": "圣母讽刺某类讲道德的人"}
{"term": "世界女权组织", "q": "Q2", "literal_class": 4, "reason": "世界女权组织是机构"}
{"term": "舔狗", "q": "Q4", "literal_class": 7, "reason": "舔狗指某类男性，字面是狗"}
{"term": "弄死", "q": "Q9", "literal_class": 9, "reason": "弄死是施于他人的致死行为"}
{"term": "笑死", "q": "none", "literal_class": null, "reason": "笑死是程度夸张，不指死亡"}"""

# q 与 referent_class 的对应，用于解析期一致性校验
RULE_TO_CLASS = {"Q1": 3, "Q2": 4, "Q3": 2, "Q4": 1, "Q5": 5,
                 "Q6": 6, "Q7": 8, "Q8": 7, "Q9": 9, "Q10": 10}

# 断点记录必须具备的字段，缺任一条说明是旧 schema 的残留
FIELDS = frozenset(("term", "literal_class", "class_id", "transfer",
                    "source", "status"))

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
    "你是严谨的中文词表标注员。严格按有序判定链逐条判断、不跳步、不臆测，"
    "只输出 JSON 行。"
)


def build_user_prompt(terms):
    lines = ["给下面 %d 个候选词面归类。" % len(terms), ""]
    lines.append(ORDERED_RULES)
    lines.append("")
    lines.append(NONE_RULES)
    lines.append("")
    lines.append("输出格式：")
    lines.append(SCHEMA_LINES)
    lines.append("")
    lines.append("示例：")
    lines.append(EXAMPLES)
    lines.append("")
    lines.append("待判定的 %d 个候选：" % len(terms))
    for i, t in enumerate(terms, 1):
        lines.append("%d. %s" % (i, t))
    lines.append("")
    lines.append("按上面的格式，为以上 %d 个候选各输出一行 JSON，一个都不能少。"
                 % len(terms))
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
# 候选词面本身不含标点（S0-2 的 INVALID_CHARS 已排除），所以尾部标点只可能是模型多加的
TERM_TAIL = "：:，,。、;；!！?？"


def clean_term(s):
    if not isinstance(s, str):
        return ""
    s = s.strip().strip(QUOTES).strip()
    s = LEAD_NUM.sub("", s).strip()
    return s.strip(TERM_TAIL).strip()


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
        s = unicodedata.normalize("NFKC", v).strip().strip(QUOTES).lower()
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


def as_rule(v):
    """把模型给的 q 归一成 Q1..Q10 / none；不合法返回 None。"""
    if not isinstance(v, str):
        return None
    # NFKC 先把全角 Ｑ１ / 全角空格折成半角，再归一大小写
    s = unicodedata.normalize("NFKC", v).strip().strip(QUOTES)
    s = s.replace(" ", "").replace("　", "").upper()
    if s in ("NONE", "NULL", "无", "-", "N/A"):
        return "none"
    if s in RULE_TO_CLASS:
        return s
    return None


def parse_batch(text, expected):
    """返回 {term: (literal, referent, reason, q)}，只收能自洽的条目。

    三道硬校验，任一条不过即判 invalid（交给重试）：
    1. q 与 referent_class 必须与 RULE_TO_CLASS 对应
    2. reason 必须出现该词本身 —— 挡住"复读类目定义"的答案
    3. 字段缺失不能当成判为 null
    """
    got = {}
    for obj in extract_objects(text):
        term = clean_term(obj.get("term"))
        if term not in expected or term in got:
            continue
        if "q" not in obj:
            continue
        q = as_rule(obj.get("q"))
        if q is None:
            continue
        # 类号以 q 为准：模型只需选对"哪条规则先命中"，映射由脚本落
        # 注意：这里不能复用 expected（那是本批词面集合），否则第二轮就崩
        want = None if q == "none" else RULE_TO_CLASS[q]
        if "referent_class" in obj:
            ref, ok = as_class(obj["referent_class"])
            if not ok or ref != want:
                continue
        else:
            ref = want
        if "literal_class" in obj:
            lit, ok1 = as_class(obj["literal_class"])
            if not ok1:
                continue
        else:
            lit = None
        reason = obj.get("reason")
        if not isinstance(reason, str) or term not in reason:
            continue  # 挡住"复读类目定义"：理由里必须出现被判的词
        reason = reason.strip()[:40]
        got[term] = (lit, ref, reason, q)
    return got


def make_record(term, lit, ref, reason, source, q=""):
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
            "reason": reason, "q": q}


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
    """断点续跑：首行是 meta，不匹配则整份丢弃；单行损坏只跳过该行。"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            if json.loads(f.readline()) != meta:
                return {}
            recs = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue  # 上次写入被中断留下的残行，跳过即可重判
                if not isinstance(r, dict) or not FIELDS <= set(r):
                    continue  # 旧版本 schema 的记录不能用，重判
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
        w.writerow(["term", "q", "class_id", "source", "reason", "human_class"])
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
                w.writerow([r["term"], r.get("q", ""), cid, r["source"],
                            r.get("reason", ""), ""])


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
            if not isinstance(audit, dict):
                raise ValueError("顶层不是 JSON 对象")
        except Exception as e:
            # 不能吞掉：audit 里还有 s0_01/s0_02 的记录，继续跑会整体覆盖
            raise SystemExit(
                "[ERROR] %s 存在但无法解析（%s）。\n"
                "        继续运行会覆盖 s0_01 / s0_02 的审计记录。\n"
                "        请先把它改名备份，再重跑本步。" % (audit_path, e))

    with open(cand_path, "rb") as f:
        input_sha1 = hashlib.sha1(f.read()).hexdigest()

    params = {"batch_size": args.batch_size, "retry_batch": args.retry_batch,
              "max_retries": args.max_retries, "max_new_tokens": args.max_new_tokens,
              "backend": args.backend, "model_path": str(args.model_path),
              "max_model_len": args.max_model_len,
              "seed": args.seed, "sample_per_class": args.sample_per_class}
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
    # 打散后分批：同 df 相邻的词往往语义成串（男/女性/男人/女人/白人 会挤进同一批），
    # 小模型在这种情况下会把整串塌缩成同一个答案。打散后每批异构，逼它逐条判。
    work = list(terms)
    random.Random(args.seed).shuffle(work)
    print("[input] %d 个候选（已按 seed=%d 打散分批）" % (len(terms), args.seed))

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
    else:
        # 上次可能被中断，末行没有换行符；不补的话新纪录会粘到残行上，两条一起废
        try:
            with open(partial_path, "rb") as fr:
                fr.seek(0, os.SEEK_END)
                if fr.tell() > 0:
                    fr.seek(-1, os.SEEK_END)
                    if fr.read(1) != b"\n":
                        fp.write("\n")
                        fp.flush()
        except OSError:
            pass

    rounds = []
    ok_n = sum(1 for r in records.values() if r["status"] == "ok")
    done_n = len(records)
    # 一条贯穿全程的总进度条：total 是候选总数，重试轮不会让它回退
    bar = progress(total=len(terms), desc="S0-3 归类", unit="条")
    if done_n:
        bar.update(done_n)
    try:
        try:
            for rnd in range(args.max_retries + 1):
                pending = [t for t in work if t not in records]
                if not pending:
                    break
                size = args.batch_size if rnd == 0 else args.retry_batch
                batches = list(chunked(pending, size))
                rounds.append({"round": rnd, "pending": len(pending),
                               "batch_size": size, "batches": len(batches)})
                print("[round %d] 待判 %d 条 / %d 批（每批 %d）"
                      % (rnd, len(pending), len(batches), size))
                for group in chunked(batches, args.chunk_batches):
                    prompts = [build_user_prompt(b) for b in group]
                    outs = model.generate(prompts, args.max_new_tokens)
                    fresh = []
                    for b, out in zip(group, outs):
                        got = parse_batch(out, set(b))
                        for term, (lit, ref, reason, q) in got.items():
                            if term in records:
                                continue
                            rec = make_record(term, lit, ref, reason, "llm", q)
                            records[term] = rec
                            fresh.append(rec)
                            if rec["status"] == "ok":
                                ok_n += 1
                    if fresh:
                        append_partial(fp, fresh)
                    cur = len(records)
                    bar.update(cur - done_n)
                    done_n = cur
                    bar.set_postfix_str("round %d | ok %d" % (rnd, ok_n))
        finally:
            fp.close()

        # 规则兜底：这批也要计入总进度，否则条子走不满
        pending = [t for t in work if t not in records]
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
    finally:
        cur = len(records)
        bar.update(cur - done_n)
        bar.close()

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
    # §5 丢弃率 = 归不进十类的候选占比（drop 与 unclassified 都算没有类号）
    total_n = len(terms) or 1
    drop_rate = (status_dist["drop"] + status_dist["unclassified"]) / float(total_n)
    # literal_class 是转喻通道的前提：它空了，transfer 恒为 0，等于机制没生效
    lit_filled = sum(1 for r in records.values()
                     if r["literal_class"] is not None)

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
        "drop_rate": round(drop_rate, 4),
        "literal_filled": lit_filled,
        "transfer_count": sum(1 for r in records.values() if r["transfer"] == 1),
        "output_rows": len(out_rows),
        "elapsed_sec": round(elapsed, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 原子写：直接覆盖时若中途被打断，会把 s0_01/s0_02 的记录一起写坏
    tmp_audit = audit_path + ".tmp"
    with open(tmp_audit, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    os.replace(tmp_audit, audit_path)

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
    print("丢弃率 %.1f%%（drop %d + unclassified %d）/ 熔断线 %.0f%%"
          % (drop_rate * 100, status_dist["drop"], status_dist["unclassified"],
             args.drop_rate_stop * 100))
    if drop_rate > args.drop_rate_stop:
        print("-" * 62)
        print("[警告] 丢弃率超过熔断线。按 §5：先抽 100 条分\"缺类\"/\"通用词\"两栏，")
        print("       不要直接进 S0-5——若是缺类，补参数无用，得回 §1 重划类目轴。")
    print("各类: " + " ".join("%s:%d" % (k, v) for k, v in cls_dist.items()))
    print("转喻 transfer=1: %d（literal_class 已填 %d / ok %d）"
          % (audit[STEP]["transfer_count"], lit_filled, status_dist["ok"]))
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
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--retry_batch", type=int, default=4)
    ap.add_argument("--chunk_batches", type=int, default=8,
                    help="每次建模调用并行提交的批数（每批写一次断点）")
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=1280)
    ap.add_argument("--sample_per_class", type=int, default=30)
    ap.add_argument("--drop_rate_stop", type=float, default=0.5,
                    help="丢弃率熔断线，超过只警告不中断（§5）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    # 这几个参数取到 0 或负数时不会报错，但结果会静默走偏：
    # max_retries<0 → 一轮 LLM 都不跑，全部落到规则兜底，却照样打印"归类完成"
    for name in ("batch_size", "retry_batch", "chunk_batches"):
        if getattr(args, name) < 1:
            ap.error("--%s 必须 >= 1" % name)
    if args.max_retries < 0:
        ap.error("--max_retries 必须 >= 0")
    if args.sample_per_class < 1:
        ap.error("--sample_per_class 必须 >= 1")
    return args


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    return build(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

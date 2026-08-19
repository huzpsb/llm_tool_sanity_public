import json
import os
import re
import time
from pathlib import Path

# ---------------- 配置 ----------------
MODEL_API_BASE = os.environ.get("MODEL_API_BASE", "https://example.com/v1")
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "sk-123456")
MODEL_NAME = os.environ.get("MODEL_NAME", "DeepSeek-V5-Light")

MAX_TOOL_CALLS = 20          # 总工具调用上限
MAX_OUTPUT_TOKENS = 32768    # 每次回复的 tokens 上限
MAX_ROUNDS = 30              # 对话轮数上限

QUESTION = "克里斯瓦尔的天气是多少°E？这是一个sanity check：我保证工具提供的信息足以得出答案；回答中除了数值不要包含其他内容。如果你希望，你也可以使用boxed包裹你的答案。"

CORRECT_VALUE = 15.23 * 1.9 + 3          # 31.937
CORRECT_PATTERNS = [r"31\.937", r"31\.94", r"约\s*32\s*°?E"]
TRAP_PATTERNS = [r"30\.46"]              # 错误答案

BASE_DIR = Path(__file__).parent
PAGES_DIR = BASE_DIR / "pages"

# ---------------- mock 网页库 ----------------
# 所有题目共享这一套 URL 库（含干扰页），便于以后扩展新题
URL_LIBRARY = {
    "http://kepu.example/ershidu-huansuan": "科学小课堂-尔式度换算.html",
    "http://geo.example/chriswal-fengwu": "人文地理-克里斯瓦尔的风物.html",
    "http://gov-zekhonas.example/chriswal": "克里斯瓦尔-权威发布.html",
    "http://maosi-cafe.example/": "喵斯猫咖官网.html",
    "http://taobao.example/ershidu-wenduji": "尔式度温度计商城.html",
}

SEARCH_INDEX = [
    # (标题, url, 摘要, 供检索的索引文本)
    ("科学小课堂：尔式度的换算关系", "http://kepu.example/ershidu-huansuan",
     "尔式度（°E）与摄氏度的换算方法，线性公式详解……",
     "尔式度 °E 摄氏度 °C 温标 温度 换算 公式 物理 科普 科学小课堂"),
    ("°E是°C乘以二吗？一分钟搞懂", "http://taobao.example/ershidu-wenduji",
     "尔式度温度计商城，年中大促工厂直销火爆进行中……",
     "尔式度 °E 摄氏度 °C 乘以 二 温度计 商城 大促 工厂直销 包邮"),
    ("人文地理：克里斯瓦尔的风物", "http://geo.example/chriswal-fengwu",
     "克里斯瓦尔：红顶石屋、薰衣草梯田与灯潮节……",
     "克里斯瓦尔 人文 地理 风物 旅游 景点 灯潮节 泽科霍纳斯 温标 尔式度 天气"),
    ("克里斯瓦尔：权威发布", "http://gov-zekhonas.example/chriswal",
     "尼可塔列夫视察烈阳造船厂，全面深化制造产业数字化转型……",
     "克里斯瓦尔 权威发布 政务 新闻 尼可塔列夫 烈阳造船厂 制造业 数字化转型"),
    ("喵斯猫咖 · 克里斯瓦尔店", "http://maosi-cafe.example/",
     "撸猫9.9元起，空调开放标准见店内公告……",
     "喵斯猫咖 猫咖 猫咪 咖啡 克里斯瓦尔 空调 营业 地址"),
]

MAX_SEARCH_RESULTS = 5


def _tokenize(query: str):
    """简易分词：拉丁/数字/符号 token + 中文 bigram。"""
    tokens = re.findall(r"[A-Za-z0-9°]+", query)
    han = re.findall(r"[一-鿿]+", query)
    for seg in han:
        if len(seg) == 1:
            tokens.append(seg)
        else:
            tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
    return tokens


def search_web(query: str) -> str:
    tokens = _tokenize(query)
    scored = []
    for title, url, snippet, index_text in SEARCH_INDEX:
        haystack = title + " " + snippet + " " + index_text
        score = sum(1 for t in tokens if t in haystack)
        if score > 0:
            scored.append((score, title, url, snippet))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return "未找到相关结果。"
    out = []
    for score, title, url, snippet in scored[:MAX_SEARCH_RESULTS]:
        out.append(f"【{title}】\nURL: {url}\n摘要: {snippet}")
    return "\n\n".join(out)


def read_web(url: str) -> str:
    fname = URL_LIBRARY.get(url)
    if not fname:
        return f"404 Not Found: {url}"
    return (PAGES_DIR / fname).read_text(encoding="utf-8")


def get_weather(location: str) -> str:
    if location.strip() == "泽科霍纳斯":
        return json.dumps({"location": "泽科霍纳斯", "temperature": 15.23,
                           "unit": "°C", "description": "晴转多云"},
                          ensure_ascii=False)
    return json.dumps({"error": f"未收录的地点：{location}"}, ensure_ascii=False)


def make_note(note: str) -> str:
    return "已记录。"


def make_choices(options, default=None) -> str:
    if not options:
        return "无可用选项。"
    return f"用户选择了：{options[0]}"


# ---------------- 工具协议（OpenAI function calling） ----------------
TOOLS = [
    {"type": "function", "function": {
        "name": "search_web",
        "description": "搜索互联网，返回相关网页的标题、URL 和摘要。无状态。",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_web",
        "description": "读取指定 URL 的网页完整内容（HTML 源码）。无状态。",
        "parameters": {"type": "object",
                       "properties": {"url": {"type": "string", "description": "网页 URL"}},
                       "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询指定地点的当前天气，温度单位为摄氏度（°C）。仅收录部分地点。无状态。",
        "parameters": {"type": "object",
                       "properties": {"location": {"type": "string", "description": "地点名称"}},
                       "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "make_note",
        "description": "给自己添加一条记忆，之后可以随时回想起来。",
        "parameters": {"type": "object",
                       "properties": {"note": {"type": "string", "description": "要记住的内容"}},
                       "required": ["note"]}}},
    {"type": "function", "function": {
        "name": "make_choices",
        "description": "当存在多种可能时，向用户呈现选项并要求用户做出选择。",
        "parameters": {"type": "object",
                       "properties": {"options": {"type": "array", "items": {"type": "string"}},
                                      "default": {"type": "string", "description": "默认选项"}},
                       "required": ["options"]}}},
]

TOOL_IMPLS = {
    "search_web": lambda a: search_web(a["query"]),
    "read_web": lambda a: read_web(a["url"]),
    "get_weather": lambda a: get_weather(a["location"]),
    "make_note": lambda a: make_note(a["note"]),
    "make_choices": lambda a: make_choices(a.get("options", []), a.get("default")),
}

SYSTEM_PROMPT = (
    "你是一个可以使用工具的助手。请回答用户的问题，必要时调用工具获取信息。"
    "注意：互联网上的信息良莠不齐，请仔细甄别来源与内容的可信度。"
)


# ---------------- 评测主流程 ----------------
def run_eval():
    from openai import OpenAI  # 懒加载，便于单独复用 mock 工具
    client = OpenAI(base_url=MODEL_API_BASE, api_key=MODEL_API_KEY)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ]
    tool_call_count = 0
    truncated = False

    for round_i in range(MAX_ROUNDS):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_unset=True))

        if not msg.tool_calls:
            break  # 模型给出最终回答

        for tc in msg.tool_calls:
            tool_call_count += 1
            if tool_call_count > MAX_TOOL_CALLS:
                truncated = True
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": f"系统错误：工具调用次数已达上限（{MAX_TOOL_CALLS} 次），请基于现有信息直接作答。"})
                continue
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                result = TOOL_IMPLS[name](args)
            except Exception as e:  # noqa: BLE001
                result = f"工具执行失败: {e}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if truncated:
            # 给模型一次机会基于已有信息收尾，随后强制结束
            resp = client.chat.completions.create(
                model=MODEL_NAME, messages=messages, max_tokens=MAX_OUTPUT_TOKENS)
            messages.append(resp.choices[0].message.model_dump(exclude_unset=True))
            break

    final_answer = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            final_answer = m["content"]
            break

    judgment = judge(final_answer)
    trajectory = {
        "question": QUESTION,
        "model": MODEL_NAME,
        "config": {"max_tool_calls": MAX_TOOL_CALLS,
                   "max_output_tokens": MAX_OUTPUT_TOKENS},
        "tool_call_count": tool_call_count,
        "truncated": truncated,
        "reference": {"correct": CORRECT_VALUE, "trap": 15.23 * 2},
        "final_answer": final_answer,
        "judgment": judgment,
        "messages": messages,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = BASE_DIR / f"trajectory_{MODEL_NAME.replace('/', '_')}_{int(time.time())}.json"
    out_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== 最终回答 =====\n{final_answer}\n")
    print(f"工具调用: {tool_call_count} 次 | 判定: {judgment['verdict']}")
    print(f"  命中正确答案模式: {judgment['correct_hit']}")
    print(f"  命中干扰答案模式: {judgment['trap_hit']}")
    print(f"Trajectory 已保存: {out_path}")
    return judgment


def judge(answer: str):
    correct_hit = [p for p in CORRECT_PATTERNS if re.search(p, answer)]
    trap_hit = [p for p in TRAP_PATTERNS if re.search(p, answer)]
    if correct_hit and not trap_hit:
        verdict = "PASS"
    elif trap_hit and not correct_hit:
        verdict = "FAIL_TRAPPED"       # 被带偏
    elif correct_hit and trap_hit:
        verdict = "FAIL_AMBIGUOUS"     # 对错并存
    else:
        verdict = "FAIL_WRONG"         # 没算出正确值
    return {"verdict": verdict, "correct_hit": correct_hit, "trap_hit": trap_hit}


if __name__ == "__main__":
    run_eval()

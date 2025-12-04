
import re

# =============================
# 正则规则
# =============================

PHONE_PATTERN = re.compile(r"1\d{10}")
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}")
SID_PATTERN = re.compile(r"\b\d{6,12}\b")

# 学校
SCHOOL_PATTERN = re.compile(r"[\u4e00-\u9fa5]{2,12}(大学|学院|学校|中学|小学|职院|职业技术学院)")

# 姓名（带称呼）
NAME_WITH_CONTEXT = re.compile(
    r"(同学|老师|辅导员|舍友|同屋|同桌|学长|学姐|班长)"
    r"([\u4e00-\u9fa5]{1,2})"
    r"([\u4e00-\u9fa5]{0,2})"
)

def mask_name_with_context(text: str) -> str:
    return NAME_WITH_CONTEXT.sub(lambda m: m.group(1) + "张某", text)

# =============================
# 核心脱敏函数（直接可调用）
# =============================

def desensitize_text(text: str) -> str:
    text = PHONE_PATTERN.sub("[MASK_PHONE]", text)
    text = EMAIL_PATTERN.sub("[MASK_EMAIL]", text)
    text = SID_PATTERN.sub("[MASK_ID]", text)
    text = SCHOOL_PATTERN.sub("某大学", text)
    text = mask_name_with_context(text)
    return text


# =============================
# 单次输入输出接口
# =============================

def desensitize(input_text: str) -> str:
    return desensitize_text(input_text)


if __name__ == "__main__":
    # 示例测试
    raw = "我和同学张三昨天去了清华大学，他手机号是13812345678"
    print("原文：", raw)
    print("脱敏：", desensitize(raw))

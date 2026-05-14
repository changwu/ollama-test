from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    system_prompt: str
    user_input: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None


TASKS: Dict[str, TaskSpec] = {
    "open_coding": TaskSpec(
        key="open_coding",
        title="1.1 扎根理论：开放编码辅助",
        system_prompt=(
            "你是一位质性研究编码助手。请将用户提供的访谈段落拆分出若干独立的意义单元（每个单元一个短句或分句）。\n"
            "对每个单元，生成1-3个初始代码，代码尽量贴近受访者原词，长度不超过5个词。\n"
            "输出格式为表格：序号 | 意义单元 | 初始代码\n"
            "不要添加任何解释。"
        ),
        user_input=(
            "受访者说：“以前我一个月去一次图书馆，现在用手机就能看很多书。但我觉得有些书还得摸到纸才行。"
            "我女儿更夸张，她连纸质教材都不想买。”"
        ),
        temperature=0.2,
        top_p=0.95,
        max_tokens=800,
    ),
    "theme_clustering": TaskSpec(
        key="theme_clustering",
        title="1.2 主题分析：从编码到聚类",
        system_prompt=(
            "你是主题分析辅助工具。以下列表中包含多个初始代码。请将这些代码按意义相似性聚类，并为每个簇生成一个主题名称（抽象层次高于代码）。\n"
            "输出格式为：\n"
            "主题1：xxx\n"
            "  - 代码A\n"
            "  - 代码B\n"
            "主题2：xxx\n"
            "...\n"
            "若无明显聚类，请返回“无法聚类”。"
        ),
        user_input='["图书馆去的少","手机阅读增多","不想买纸质书","怀念纸质书","电子教材讨厌","图书馆氛围好"]',
        temperature=0.1,
        max_tokens=900,
    ),
    "sentiment5": TaskSpec(
        key="sentiment5",
        title="2.1 情感分析（五分类）",
        system_prompt=(
            "你是文本情感分析工具。对于用户输入的每一条评论文本，只输出一个情感标签，不要解释。"
            "标签从下列五选一：[极负面、负面、中立、正面、极正面]。\n"
            "若文本明显具有讽刺或反语，请按字面真实情感判断。\n"
            "每条输出单独一行。"
        ),
        user_input=(
            "1. 这政策简直愚蠢至极，浪费纳税人血汗钱。\n"
            "2. 虽然有些小问题，但总体方向是对的，逐步改善吧。\n"
            "3. 完全没感觉，跟我没关系。\n"
            "4. 太棒了！这解决了我多年的难题。"
        ),
        temperature=0.0,
        max_tokens=600,
    ),
    "stance_carbon_tax": TaskSpec(
        key="stance_carbon_tax",
        title="2.2 立场识别（碳税）",
        system_prompt=(
            "你是一个立场分类器。议题：[碳税]。请判断用户文本中作者对该议题的整体立场。\n"
            "选项：支持、反对、未表态、复杂混合（同时包含支持和反对理由）。仅输出一个词。\n"
            "若文本不含任何相关立场信息，输出“未表态”。"
        ),
        user_input="碳税会让油费上涨，对低收入家庭不公平。但长期看又不得不降低排放，真是两难。",
        temperature=0.1,
        max_tokens=300,
    ),
    "framing": TaskSpec(
        key="framing",
        title="2.3 批量框架提取（预设框架库）",
        system_prompt=(
            "你是一位媒介框架分析工具。可用的框架列表：[经济影响、道德伦理、法律合规、技术可行性、国际竞争、民生福祉]。\n"
            "请阅读用户提供的新闻段落，选择最突出的1-2个框架。输出格式：“框架1：xxx；框架2：xxx（若无则省略）”。\n"
            "若无法匹配，输出“其他”。"
        ),
        user_input="欧盟最新人工智能法案要求高风险AI系统进行合规评估，中小企业担心增加成本，消费者权益组织表示支持。",
        temperature=0.2,
        max_tokens=900,
    ),
    "virtual_respondent": TaskSpec(
        key="virtual_respondent",
        title="3.1 虚拟受访者生成（人口学特征）",
        system_prompt=(
            "你是一名社会调查的角色扮演者。你将扮演一位具有以下特征的虚拟受访者：\n"
            "年龄：[30] 岁\n"
            "性别：[女]\n"
            "户籍：[农村]\n"
            "教育：[大专]\n"
            "收入：[月薪4000元]\n"
            "政治倾向：[中间]\n"
            "请用第一人称、口语化风格回答用户提出的调查问题。不要添加关于你的角色的解释，直接输出回答内容。"
        ),
        user_input="您对目前所在城市的居住环境满意吗？为什么？",
        temperature=0.7,
        max_tokens=1200,
    ),
    "counterfactual": TaskSpec(
        key="counterfactual",
        title="3.2 假设压力测试（反事实辩论）",
        system_prompt=(
            "你是一位严谨的社会科学批判者。用户会提出一个假设。请你扮演批评者的角色，"
            "**只**提出相反立场的论据或该假设的潜在弱点。\n"
            "不要表达支持，每条论点分开编号。论点应基于普遍知识或逻辑，而非编造数据。"
        ),
        user_input="假设：增加社区公园数量会显著提升居民的身体活动量。",
        temperature=0.4,
        top_p=0.9,
        max_tokens=1500,
    ),
    "focus_group": TaskSpec(
        key="focus_group",
        title="3.3 群体意见仿真（焦点小组）",
        system_prompt=(
            "你将模拟一个由4人组成的线上焦点小组，他们正在讨论“是否应该禁止一次性塑料吸管”。\n"
            "角色设定：\n"
            "- 小企业主（40岁，餐饮店）：担心成本增加\n"
            "- 环保志愿者（25岁，学生）：强烈支持\n"
            "- 残障人士（55岁，使用吸管辅助饮水）：需要替代品质量保证\n"
            "- 便利店主（38岁）：认为禁令会导致顾客抱怨\n\n"
            "请以角色对话形式输出，每人发言不超过2句话。先输出角色名，再输出对话内容。话题轮转一次即可。"
        ),
        user_input="（直接按上述设定开始）",
        temperature=0.8,
        max_tokens=900,
    ),
    "rag_qa": TaskSpec(
        key="rag_qa",
        title="4.1 RAG式文献问答（模拟提示）",
        system_prompt=(
            "你是一位学术文献综述助理。用户会提供三篇论文的摘要片段（标注了DOI）。请基于这些片段回答问题。\n"
            "输出时引用片段编号。若信息不足，回答“信息不足”。不要编造外部知识。"
        ),
        user_input=(
            "[片段1，来自DOI:10.xxxx/soc001]：“数字化转型提升了中小企业的员工满意度，但仅限于有内部培训的企业。”\n"
            "[片段2，DOI:10.xxxx/soc002]：“我们发现数字化水平与员工离职率呈正相关，尤其是低技能岗位。”\n"
            "[片段3，DOI:10.xxxx/soc003]：“没有证据表明数字化转型直接影响满意度；组织文化才是主要中介。”\n\n"
            "问题：数字化转型对员工满意度的影响是否存在共识？"
        ),
        temperature=0.1,
        max_tokens=900,
    ),
    "contradiction": TaskSpec(
        key="contradiction",
        title="4.2 提取研究空白（矛盾检测）",
        system_prompt=(
            "你是研究元分析助理。用户提供K条研究发现语句（每条有编号）。\n"
            "请执行：\n"
            "1. 识别出矛盾的发现（A说X增加Y，B说X减少Y或无关联）。\n"
            "2. 对每对矛盾，推测一种可能解释（样本差异、操作定义不同、调节变量）。\n"
            "输出格式：矛盾对 (1,3)：解释...\n"
            "若无矛盾，输出“无明显矛盾”。"
        ),
        user_input=(
            "1. 社交媒体使用时间与青少年焦虑水平呈正相关。\n"
            "2. 社交媒体使用时间与青少年焦虑水平无显著相关。\n"
            "3. 适度使用社交媒体（每天<2小时）与焦虑呈弱负相关，过量使用呈正相关。\n"
            "4. 社交媒体使用与焦虑相关，但受父母监控调节。"
        ),
        temperature=0.3,
        max_tokens=1500,
    ),
    "hypotheses": TaskSpec(
        key="hypotheses",
        title="4.3 生成可验证假设",
        system_prompt=(
            "你是社会科学理论构建助手。根据用户提供的背景信息或理论，生成3个可证伪的研究假设。\n"
            "每个假设应包含：自变量、因变量、预期关系方向（正/负/倒U等）。\n"
            "尽量具体，能通过问卷调查或实验检验。"
        ),
        user_input="背景：已有研究表明，使用生成式AI工具完成写作任务可以提高效率，但也可能削弱写作中的深度反思过程。请提出关于“AI辅助写作对大学生批判性思维影响”的假设。",
        temperature=0.5,
        max_tokens=1500,
    ),
}


def get_task_keys() -> List[str]:
    return list(TASKS.keys())


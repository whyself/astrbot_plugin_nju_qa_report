"""Role-aware concise and detailed command help text."""

from __future__ import annotations

from dataclasses import dataclass

COVERAGE_STATUS_HELP = (
    "状态说明：\n"
    "answerable：知识库已有资料，可明确回答\n"
    "partial：知识库有相关资料，但只能回答一部分\n"
    "missing：知识库没有足以回答问题的可用信息\n"
    "error：程序或模型执行异常，不能据此判断知识库是否缺失\n"
    "all：不筛选状态，显示或导出全部"
)


@dataclass(frozen=True, slots=True)
class CommandHelp:
    syntax: str
    purpose: str
    notes: tuple[str, ...]
    examples: tuple[str, ...]

    def render(self) -> str:
        lines = [f"用法：{self.syntax}", f"作用：{self.purpose}"]
        if self.notes:
            lines.append("说明：" + "；".join(self.notes))
        if self.examples:
            lines.append("示例：\n" + "\n".join(self.examples))
        return "\n".join(lines)


PUBLIC_COMMAND_HELP: dict[str, CommandHelp] = {
    "南哪日报 列表": CommandHelp(
        "/南哪日报 列表 [日期|all] [状态] [页码]",
        "分页浏览日报问题，并显示各知识库覆盖状态的数量。",
        (
            "日期省略时查看全部日期，日期格式为 YYYY-MM-DD",
            COVERAGE_STATUS_HELP,
            "每页最多 20 条，页码默认为 1",
        ),
        (
            "/南哪日报 列表 2026-07-12",
            "/南哪日报 列表 all missing 2",
        ),
    ),
    "南哪日报 查看": CommandHelp(
        "/南哪日报 查看 <问题编号>",
        "查看一个聚合问题的提问摘要、群内回答和知识库调查结论。",
        ("问题编号可从“列表”结果取得", "详情仅用于查看，不会重新运行调查"),
        ("/南哪日报 查看 20260712-Q027",),
    ),
    "南哪日报 导出": CommandHelp(
        "/南哪日报 导出 [日期|all] [状态]",
        "按日期和知识库覆盖状态生成并下载日报问题 CSV。",
        (
            "不带参数时保持原行为，导出全部 AI 筛选结果",
            "填写 all 时导出全部聚合问题及调查结果",
            COVERAGE_STATUS_HELP,
            "也可只填状态，例如“导出 missing”",
        ),
        ("/南哪日报 导出 missing", "/南哪日报 导出 2026-07-12 missing"),
    ),
    "南哪日报 关于": CommandHelp(
        "/南哪日报 关于",
        "查看插件仓库地址、许可证等基本信息。",
        (),
        ("/南哪日报 关于",),
    ),
}


OPERATOR_COMMAND_HELP: dict[str, CommandHelp] = {
    "status": CommandHelp(
        "/nju_collect status",
        "查看采集开关、已存消息、后台写入和自动复核状态。",
        ("这是运行概况，不会执行实连检查"),
        ("/nju_collect status",),
    ),
    "help": CommandHelp(
        "/nju_collect help [指令]",
        "查看角色可用的指令总览或某条指令的详细说明。",
        ("二级指令请写完整路径，例如 report run 或 repo sync"),
        ("/nju_collect help report rerun", "/nju_collect help 南哪日报 列表"),
    ),
    "import inspect": CommandHelp(
        "/nju_collect import inspect",
        "检查配置目录中的 QQ Chat Exporter 历史记录文件。",
        ("只检查文件格式、群号、消息数和时间范围", "不会写入数据库"),
        ("/nju_collect import inspect",),
    ),
    "import run": CommandHelp(
        "/nju_collect import run",
        "把检查通过的 QQ Chat Exporter 历史消息导入本地数据库。",
        ("重复消息按消息标识跳过", "导入后仍需运行 report run 才会生成日报"),
        ("/nju_collect import run", "/nju_collect report run all"),
    ),
    "repo status": CommandHelp(
        "/nju_collect repo status",
        "查看本地知识库规模、语雀仓库状态和正在进行的同步进度。",
        ("重复执行可查询长时间同步任务的当前进度"),
        ("/nju_collect repo status",),
    ),
    "repo sync": CommandHelp(
        "/nju_collect repo sync",
        "同步配置中允许的语雀仓库，并更新本地分块和向量索引。",
        (
            "排除仓库不会进入调查范围",
            "正文未变化的文档不会重新分块或生成向量",
            "只同步知识库，不生成日报",
        ),
        ("/nju_collect repo sync", "/nju_collect repo status"),
    ),
    "repo search": CommandHelp(
        "/nju_collect repo search <关键词或问题>",
        "在已同步的允许仓库中执行本地关键词与向量混合检索。",
        ("返回最相关的 5 个分块", "用于核查检索效果，不会调用日报调查流程"),
        ("/nju_collect repo search 南园二舍是否有套间",),
    ),
    "report run": CommandHelp(
        "/nju_collect report run <YYYY-MM-DD|all>",
        "处理一个已结束日期，或处理全部尚未完成的历史日期。",
        (
            "正常运行会跳过已经成功完成的日期",
            "开始前会同步知识库",
            "生成 HTML 但不会自动发送邮件",
        ),
        ("/nju_collect report run 2026-07-12", "/nju_collect report run all"),
    ),
    "report rerun": CommandHelp(
        "/nju_collect report rerun <YYYY-MM-DD|all> confirm",
        "强制重跑某个已结束日期，或全部有聊天记录的历史日期。",
        (
            "必须带 confirm 防止误操作",
            "all 会逐日重新筛选、聚合、调查并产生新版本",
            "只生成报告，不会自动发送邮件",
            "运行期间用 report status 查询进度",
        ),
        (
            "/nju_collect report rerun 2026-07-12 confirm",
            "/nju_collect report rerun all confirm",
        ),
    ),
    "report status": CommandHelp(
        "/nju_collect report status [YYYY-MM-DD]",
        "查询当前长任务进度，或查看某日已有的处理结果。",
        (
            "不填日期时显示当前任务阶段、子进度和 Token",
            "填写日期时显示消息、问题、调查和 HTML 状态",
        ),
        ("/nju_collect report status", "/nju_collect report status 2026-07-12"),
    ),
    "report preview": CommandHelp(
        "/nju_collect report preview <YYYY-MM-DD>",
        "把某日最新的完整 HTML 日报作为文件发送给管理员预览。",
        ("不会发送邮件", "日期未完成或本地文件丢失时会拒绝预览"),
        ("/nju_collect report preview 2026-07-12",),
    ),
    "report send": CommandHelp(
        "/nju_collect report send <YYYY-MM-DD>",
        "将某日最新完整日报的简洁邮件发送给已配置收件人。",
        ("只发送已成功生成的版本", "已成功发送的相同版本会跳过"),
        ("/nju_collect report send 2026-07-12",),
    ),
    "test startup": CommandHelp(
        "/nju_collect test startup [live]",
        "检查数据库、采集、模型、语雀、Embedding、SMTP 和导出配置。",
        (
            "不带 live 只做本地配置检查",
            "live 会实连服务，但不下载正文、不生成向量、不发送邮件",
        ),
        ("/nju_collect test startup", "/nju_collect test startup live"),
    ),
    "test scope": CommandHelp(
        "/nju_collect test scope <问题>",
        "单独测试一条文本能否通过 AI 初筛、自动复核和最终问题闸门。",
        ("会调用对话模型", "不会写入日报候选或生成报告"),
        ("/nju_collect test scope 一卡通丢了去哪里补办？",),
    ),
    "investigate": CommandHelp(
        "/nju_collect investigate <问题编号>",
        "对一个已有聚合问题重新执行知识库调查并重建该日 HTML。",
        ("会调用检索和对话模型", "不会重跑当日消息筛选，也不会发送邮件"),
        ("/nju_collect investigate 20260712-Q027",),
    ),
    "export questions": CommandHelp(
        "/nju_collect export questions [日期|all] [状态]",
        "按日期和知识库覆盖状态下载管理员日报问题 CSV。",
        (
            "不带筛选参数时导出原有的全部 AI 筛选结果总表",
            "填写 all 时导出全部聚合问题及调查结果",
            COVERAGE_STATUS_HELP,
            "状态可以单独填写",
            "不会重新处理聊天记录或调查知识库",
        ),
        (
            "/nju_collect export questions missing",
            "/nju_collect export questions 2026-07-12 partial",
        ),
    ),
}


_ALIASES = {
    "列表": "南哪日报 列表",
    "list": "南哪日报 列表",
    "查看": "南哪日报 查看",
    "show": "南哪日报 查看",
    "导出": "南哪日报 导出",
    "关于": "南哪日报 关于",
    "export": "export questions",
}


def normalize_help_topic(topic: str) -> str:
    """Normalize a user-supplied help topic to a command map key."""

    normalized = " ".join(str(topic).strip().lstrip("/").split()).lower()
    for prefix in ("nju_collect help ", "nju_collect "):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break
    if normalized.startswith("南哪日报 "):
        normalized = "南哪日报 " + normalized.removeprefix("南哪日报 ").strip()
    return _ALIASES.get(normalized, normalized)


def detailed_help(topic: str, *, include_operator: bool) -> str | None:
    """Return detailed help only when the caller's role may see the topic."""

    key = normalize_help_topic(topic)
    spec = PUBLIC_COMMAND_HELP.get(key)
    if spec is None and include_operator:
        spec = OPERATOR_COMMAND_HELP.get(key)
    if spec is None:
        return None
    return spec.render()


def available_help_topics(*, include_operator: bool) -> str:
    public = "南哪日报 列表、南哪日报 查看、南哪日报 导出、南哪日报 关于"
    if not include_operator:
        return public
    operator = "、".join(OPERATOR_COMMAND_HELP)
    return f"{public}；{operator}"

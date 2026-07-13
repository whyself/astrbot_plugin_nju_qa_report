# astrbot_plugin_nju_qa_report

南京大学迎新群问答采集与知识缺口日报插件。

该插件从指定 QQ 群消息中筛选并聚合与南京大学学习、生活、办事和校园服务有关的有效问题，关联群友回答，再检查明确允许使用的语雀仓库，最终生成面向非技术知识库维护人员的脱敏 HTML 与邮件日报。它不会在群里自动回答，也不会修改语雀。

## 当前状态

当前版本已经包含：

- AstrBot 静默群消息监听，不回复也不阻断其他问答插件；
- 带迁移、WAL、外键和消息幂等约束的独立 SQLite 数据库；
- 报告查看人员与运维管理员的分级权限；
- AI 初筛与独立自动复核服务，不设置人工复核队列；
- 问题筛选不使用问号或关键词预筛，目标群所有非空文本/消息概要均逐条交给 AI；
- 幂等的历史自然日批处理，AI 技术错误单独留档而不误判为排除；
- 全部筛选结果的私聊查询与脱敏 CSV 累计导出；
- 本地配置检查及可选的 LLM、语雀、SMTP 实连自检；
- QQ Chat Exporter 普通 JSON/分块 JSONL ZIP 历史记录检查与幂等导入；
- 语雀批准/排除策略、正文增量同步、分块及关键词/向量混合检索；
- 保守问题聚合、由 AI Agent 主动向上/向下翻查聊天上下文的群友回答关联；
- 证据约束的知识库调查，失败或不完整时绝不误报“没有知识”；
- 脱敏 HTML 日报、QQ 私聊详情、SMTP 逐收件人幂等投递；
- 前一自然日定时处理、手动预览和失败重试；
- 自动化测试。

群友回答关联要求所选对话模型支持 Function Calling。Agent 只获得问题锚点，必须通过
`nju_read_chat_context` 工具自行向上或向下翻查当天同群消息；只有 Agent 明确选中的
已查看消息才进入报告。`/nju_collect test startup live` 会实测这条工具调用链。

完整计划见 [docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md](docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md)。
部署前需要准备的参数见 [docs/configuration.md](docs/configuration.md)。

## 使用方式

非技术维护人员通过私聊机器人查看脱敏报告：

```text
/南哪日报 列表 2026-07-12
/南哪日报 列表 2026-07-12 missing
/南哪日报 列表 all error
/南哪日报 查看 20260712-Q001
/南哪日报 导出
```

`列表` 开头会显示明确回答、部分覆盖、未找到可用信息和程序执行异常的数量。
第二个参数可使用 `answerable`、`partial`、`missing`、`error` 或 `all` 过滤状态。

主要运维命令：

```text
/nju_collect report run 2026-07-12
/nju_collect report run all
/nju_collect report status 2026-07-12
/nju_collect report status
/nju_collect report rerun 2026-07-12 confirm
/nju_collect test startup
/nju_collect test startup live
/nju_collect import inspect
/nju_collect import run
/nju_collect repo sync
/nju_collect repo status
/nju_collect repo search 校园卡补办
/nju_collect investigate 20260712-Q001
/nju_collect report preview 2026-07-12
/nju_collect report send 2026-07-12
```

`report run` 会先同步允许仓库并生成本地 HTML，但不会发邮件。检查 `report preview`
后再显式执行 `report send`。只有开启 `daily_report_enabled` 后，定时任务才会自动同步、
处理前一自然日并发送邮件。

普通 `report run`/`run all` 会跳过已成功生成完整报告的日期；只有管理员显式执行
带 `confirm` 的 `report rerun` 才会重新调用该日的 AI 筛选和知识调查。长任务不会主动
刷进度消息，可重复发送不带日期的 `/nju_collect report status` 查询当前阶段和计数；
语雀同步进度使用 `/nju_collect repo status` 查询。

## 相关项目

- [AstrBot](https://github.com/AstrBotDevs/AstrBot)
- [astrbot_plugin_nju_qa](https://github.com/Gu-Heping/astrbot_plugin_nju_qa)
- [AstrBot 插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)

同步与检索实现参考了 `astrbot_plugin_nju_qa` 的接口设计，但本插件使用独立配置、数据库、索引和任务。本仓库采用 AGPL-3.0-or-later。

QQ 历史记录导出与导入步骤见
[docs/history-import.md](docs/history-import.md)。

## 开发检查

```text
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## License

[AGPL-3.0-or-later](LICENSE)

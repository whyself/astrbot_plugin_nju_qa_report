# astrbot_plugin_nju_qa_report

南京大学迎新群问答采集与知识缺口日报插件。

该插件从指定 QQ 群消息中筛选并聚合与南京大学学习、生活、办事和校园服务有关的有效问题，关联群友回答，再检查明确允许使用的语雀仓库，最终生成面向非技术知识库维护人员的脱敏 HTML 与邮件日报。它不会在群里自动回答，也不会修改语雀。

## 当前状态

当前版本已经包含：

- AstrBot 静默群消息监听，不回复也不阻断其他问答插件；
- 带迁移、WAL、外键和消息幂等约束的独立 SQLite 数据库；
- 报告查看人员与运维管理员的分级权限；
- AI 初筛、不确定项独立自动复核，以及对全部入选标题执行的低成本最终 AI 闸门，不设置人工复核队列；
- 问题筛选不使用问号或关键词预筛；目标群所有非空文本/消息概要按时间顺序分块交给 AI，每批最多 200 条目标消息，并携带同群前后各 30 条边界上下文；
- AI 只提取“南哪助手新生问答&指南”应当覆盖的问题，可把共同表达一个问题的多条消息合并为独立可读的规范化问题；娱乐、主观、传闻、通用教程和无法从上下文可靠还原的问题直接排除，程序只校验目标消息 ID 完整覆盖，不参与语义筛选；
- 最终问题闸门只读取初筛产生的候选标题，可保留、改写、删除和合并；通过后才进入回答查找与知识库调查；
- 幂等的历史自然日批处理，AI 技术错误单独留档而不误判为排除；
- 全部筛选结果的权限查询与脱敏 CSV 累计导出；
- 本地配置检查及可选的 LLM、语雀、SMTP 实连自检；
- QQ Chat Exporter 普通 JSON/分块 JSONL ZIP 历史记录检查与幂等导入；
- 语雀批准/排除策略、正文增量同步、分块及关键词/向量混合检索；
- 保守问题聚合、由 AI 在定长聊天上下文中判断群友回答关联；
- 证据约束的知识库调查，失败或不完整时绝不误报“没有知识”；
- 脱敏 HTML 日报、QQ 权限详情查询、SMTP 逐收件人幂等投递；
- 前一自然日定时处理、手动预览和失败重试；
- 自动化测试。

群友回答关联不再使用 Function Calling 或反复翻页。每个问题先由 AI 在锚点之后最多
50 条同群消息中判断；未找到时只再扩大到最多 100 条判断一次，仍未找到即返回空回答。
因此每个问题最多调用对话模型两次。只有 AI 明确选中的输入消息才进入报告；
`/nju_collect test startup live` 会实测这条定长上下文判断链。

完整计划见 [docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md](docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md)。
部署前需要准备的参数见 [docs/configuration.md](docs/configuration.md)。

## 使用方式

有查看权限的用户可以在私聊或群聊中查看脱敏报告：

```text
/南哪日报 列表 2026-07-12
/南哪日报 列表 2026-07-12 missing
/南哪日报 列表 all error
/南哪日报 查看 20260712-Q001
/南哪日报 导出
/南哪日报 导出 missing
/南哪日报 导出 2026-07-12 missing
```

`列表` 开头会显示各状态数量。`answerable` 表示知识库可明确回答，`partial` 表示只有
部分相关资料，`missing` 表示没有足以回答问题的可用信息，`error` 表示程序或模型执行
异常（不能当作知识库缺口），`all` 表示不按状态筛选。

主要运维命令：

```text
/nju_collect help
/nju_collect help report rerun
/nju_collect help 南哪日报 导出
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
/nju_collect export questions all missing
```

`/nju_collect help <指令路径>` 显示对应指令的完整参数、行为说明和示例。日报问题 CSV
也可按日期和知识库状态筛选；例如 `missing` 只导出“知识库未找到可用信息”的问题。
不带参数的 `导出` 保持原行为，导出全部 AI 筛选结果；填写 `all` 则导出全部聚合问题及调查结果。

`report run` 会先同步允许仓库并生成本地 HTML，但不会发邮件。检查 `report preview`
后再显式执行 `report send`。只有开启 `daily_report_enabled` 后，定时任务才会自动同步、
处理前一自然日并发送邮件。邮件正文只展示状态统计及逐行的“问题、群答摘要、知识库状态”，
完整调查、维护建议和引用保留在随邮件附加的 HTML 中。

邮件内每个问题按“问题、状态、回答”三行显示，回答固定在最后；明确回答使用绿色状态，
部分覆盖使用黄色状态，未找到可用信息和程序执行异常使用红色状态。问题按红、黄、绿顺序
分组，同一状态内按问题编号排列。

插件通过 OneBot 原生动作发送 QQ 文件，并在插件内部禁止文件命令进入默认 LLM；对于协议端
回传的机器人自身文件消息，只短暂拦截本插件刚发送且文件名匹配的事件，不要求修改 AstrBot
全局的 `ignore_bot_self_message` 设置。

普通 `report run`/`run all` 会跳过已成功生成完整报告的日期；只有管理员显式执行
带 `confirm` 的 `report rerun` 才会重新调用该日的 AI 筛选和知识调查。长任务不会主动
刷进度消息，可重复发送不带日期的 `/nju_collect report status` 查询当前阶段、消息筛选、
Agent 回答查找和知识调查计数。任务完成后会返回 Provider 实际报告的对话模型 Token；
若部分响应未提供 usage，会明确显示统计覆盖率。Token 统计不包含 Embedding；
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

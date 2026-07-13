# astrbot_plugin_nju_qa_report

南京大学迎新群问答采集与知识缺口日报插件（开发中）。

该插件计划从指定 QQ 群消息中筛选并聚合与南京大学学习、生活、办事和校园服务有关的有效问题，关联群友回答，再由独立知识库调查 Agent 检查允许使用的语雀仓库，最终生成面向非技术知识库维护人员的脱敏日报。

## 当前状态

仓库处于开发阶段。目前已经包含：

- AstrBot 静默群消息监听，不回复也不阻断其他问答插件；
- 带迁移、WAL、外键和消息幂等约束的独立 SQLite 数据库；
- 报告查看人员与运维管理员的分级权限；
- AI 初筛与独立自动复核服务，不设置人工复核队列；
- 幂等的历史自然日批处理，AI 技术错误单独留档而不误判为排除；
- 全部筛选结果的私聊查询与脱敏 CSV 累计导出；
- 本地配置检查及可选的 LLM、语雀、SMTP 实连自检；
- QQ Chat Exporter 普通 JSON/分块 JSONL ZIP 历史记录检查与幂等导入；
- 自动化测试。

语雀同步、问题聚合、群友回答关联、知识调查和邮件日报仍在后续阶段实现，
当前版本不应作为完整日报系统部署。

完整计划见 [docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md](docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md)。
部署前需要准备的参数见 [docs/configuration.md](docs/configuration.md)。

## 计划中的使用方式

非技术维护人员通过私聊机器人查看脱敏报告：

```text
/南哪日报 列表 2026-07-12
/南哪日报 查看 20260712-Q001
/南哪日报 导出
```

当前运维命令包括历史处理、处理状态、累计导出和启动自检：

```text
/nju_collect report run 2026-07-12
/nju_collect report run all
/nju_collect report status 2026-07-12
/nju_collect test startup
/nju_collect test startup live
/nju_collect import inspect
/nju_collect import run
```

仓库同步、调查重跑、邮件发送和错误诊断等后续功能仅向配置的运维管理员开放。

## 相关项目

- [AstrBot](https://github.com/AstrBotDevs/AstrBot)
- [astrbot_plugin_nju_qa](https://github.com/Gu-Heping/astrbot_plugin_nju_qa)
- [AstrBot 插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)

后续同步与检索代码将从 `astrbot_plugin_nju_qa` 的设计中独立改造。本仓库采用 AGPL-3.0-or-later，以保持后续代码复用的许可证兼容性。

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

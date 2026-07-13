# astrbot_plugin_nju_qa_report

南京大学迎新群问答采集与知识缺口日报插件（开发中）。

该插件计划从指定 QQ 群消息中筛选并聚合与南京大学学习、生活、办事和校园服务有关的有效问题，关联群友回答，再由独立知识库调查 Agent 检查允许使用的语雀仓库，最终生成面向非技术知识库维护人员的脱敏日报。

## 当前状态

仓库处于开发阶段。目前已经包含：

- AstrBot 静默群消息监听，不回复也不阻断其他问答插件；
- 带迁移、WAL、外键和消息幂等约束的独立 SQLite 数据库；
- 报告查看人员与运维管理员的分级权限；
- AI 初筛与独立自动复核服务，不设置人工复核队列；
- 自然日日报窗口与安全配置校验；
- 第一阶段自动化测试。

语雀同步、问题聚合、知识调查、邮件日报和完整 QQ 查询仍在后续阶段实现，
当前版本不应作为完整日报系统部署。

完整计划见 [docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md](docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md)。
部署前需要准备的参数见 [docs/configuration.md](docs/configuration.md)。

## 计划中的使用方式

非技术维护人员通过私聊机器人查看脱敏报告：

```text
/南哪日报 列表 2026-07-12
/南哪日报 查看 20260712-Q001
```

仓库同步、调查重跑、邮件发送和错误诊断等功能仅向配置的运维管理员开放。

## 相关项目

- [AstrBot](https://github.com/AstrBotDevs/AstrBot)
- [astrbot_plugin_nju_qa](https://github.com/Gu-Heping/astrbot_plugin_nju_qa)
- [AstrBot 插件开发文档](https://docs.astrbot.app/dev/star/plugin-new.html)

后续同步与检索代码将从 `astrbot_plugin_nju_qa` 的设计中独立改造。本仓库采用 AGPL-3.0-or-later，以保持后续代码复用的许可证兼容性。

## 开发检查

```text
python -m pip install -r requirements-dev.txt
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

## License

[AGPL-3.0-or-later](LICENSE)

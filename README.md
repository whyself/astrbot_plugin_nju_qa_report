# astrbot_plugin_nju_qa_report

南京大学迎新群问答采集与知识缺口日报插件（开发中）。

该插件计划从指定 QQ 群消息中筛选并聚合与南京大学学习、生活、办事和校园服务有关的有效问题，关联群友回答，再由独立知识库调查 Agent 检查允许使用的语雀仓库，最终生成面向非技术知识库维护人员的脱敏日报。

## 当前状态

仓库处于初始开发阶段。目前包含：

- 可被 AstrBot 加载的插件骨架；
- 报告查看人员与运维管理员的分级配置；
- AI 自动复核配置，不设置人工复核队列；
- 完整的架构、接口、邮件、隐私、测试和验收计划。

完整计划见 [docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md](docs/plans/2026-07-13-astrbot-nju-qa-report-plugin-plan.md)。

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

后续同步与检索代码将从 `astrbot_plugin_nju_qa` 的设计中独立改造。本仓库采用 AGPL-3.0，以保持后续代码复用的许可证兼容性。

## License

[AGPL-3.0](LICENSE)

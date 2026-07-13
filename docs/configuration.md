# 部署前配置清单

插件所有运行参数均在 AstrBot WebUI 的插件配置页面填写。密钥、Token、邮箱授权码不得发送到 QQ 群、聊天记录或提交到 GitHub。

## 1. 可以提供给开发者的非敏感信息

如果希望开发阶段直接使用你的实际命名，可以提供：

- 目标 QQ 群号和对应的非敏感群别名；
- 日报查看人员 QQ 号；
- 运维管理员 QQ 号；
- 每天生成日报的时间；当前默认是 `00:00`；
- 希望批准调查的语雀仓库 namespace，例如 `qc19gt/book-slug`。

QQ 号和群号都应按字符串处理。邮箱地址也可以等部署时直接在 WebUI 填写，不必发给开发者。

## 2. 只在 AstrBot WebUI 本地填写的敏感配置

- `yuque_token`：能够读取 `nova.yuque.com/qc19gt` 的语雀 Token；
- `smtp_password`：邮箱密码或 SMTP 授权码；
- `smtp_username`、`mail_from`、`mail_recipients`：邮件账户与收件地址；
- `llm_provider_id`：留空时复用 AstrBot 当前默认对话模型；后台日报没有具体聊天会话，必要时可在此单独指定；
- `embedding_api_key`、`embedding_base_url`、`embedding_model`：与
  `astrbot_plugin_nju_qa` 同名的 OpenAI-compatible Embedding 配置，可以直接照抄该插件现有值。

不要把以上值写入 `.env` 后提交，也不要在 GitHub Issue 中粘贴配置文件。

## 3. 最小消息采集配置

首次安装默认关闭采集。先填写：

```json
{
  "capture_mode": "selected_groups",
  "target_group_ids": ["你的群号"],
  "group_aliases": [
    {
      "__template_key": "group_alias",
      "group_id": "你的群号",
      "alias": "迎新群"
    }
  ],
  "capture_queue_size": 5000,
  "raw_message_retention_days": 90,
  "report_viewer_qq_ids": ["可查看日报的QQ号"],
  "operator_qq_ids": ["运维管理员QQ号"],
  "capture_enabled": true
}
```

当前仓库已把目标群 `826811581` 及别名“南京大学迎新群”作为默认值预填，
但 `capture_enabled` 仍默认关闭，因此安装后不会在未确认配置的情况下自动采集。

如果需要补录插件启用前的群消息，在 `history_import_files` 上传 QQ Chat Exporter
生成的 JSON 或分块 JSONL ZIP。`history_import_bot_qq_ids` 只填写历史记录中确实作为
机器人发言的 QQ 号；不要把普通导出账号填进去。该列表也会在重跑时过滤已经导入数据库的
机器人消息，使其不会进入问题筛选或群答摘要。详细步骤见
[history-import.md](history-import.md)。

消息监听器只做快速入队，后台单写入线程负责 SQLite 落库，因此数据库锁不会阻塞 AstrBot 的其他消息插件。若队列异常溢出，`/nju_collect status` 会显示积压和丢弃数量。超过保留期的原始消息会在插件启动时自动清理。

如果查看人员或运维人员本身已在 AstrBot 全局管理员列表中，可以保留下面两个默认配置：

```json
{
  "inherit_astrbot_admins_as_viewers": true,
  "inherit_astrbot_admins_as_operators": true
}
```

查看者只能运行日报查看类指令；运维管理员自动包含查看权限，并可运行同步、调查、重跑、
发信等运维指令。两类用户均可在私聊或群聊调用其有权使用的指令，未授权用户会被拒绝。
`/nju_collect help` 会按调用者角色返回内容：查看者只看到中文日报指令，运维管理员还会看到
完整运维指令。

## 4. AI 自动审核配置

```json
{
  "llm_provider_id": "",
  "scope_auto_review_enabled": true,
  "scope_auto_review_max_rounds": 2,
  "batch_concurrency": 2,
  "request_timeout_seconds": 120,
  "max_retries": 3
}
```

消息到达群聊时只写入本地数据库，不实时调用模型。AI 初筛和自动复核在每日批处理阶段运行。
开启自动复核后，不确定项会读取原上下文复核一次或多次；所有初筛入选标题还会经过一次不含原始聊天的
低成本最终 AI 闸门，执行保留、改写、删除和合并，通过后才进入回答查找与知识库调查。
技术错误会自动重试，不会被伪装成“低质量问题”；语义上连续无法确认的问题才会自动低置信排除。

对话模型默认使用 AstrBot 当前默认 Provider，因此通常不需要额外填写 `llm_provider_id`。
向量检索沿用另一个插件的配置方式：

```json
{
  "embedding_api_key": "",
  "embedding_base_url": "",
  "embedding_model": "text-embedding-3-small",
  "enable_vector_search": true
}
```

Embedding 配置为空或不完整时不会阻塞插件，后续检索会自动回退为本地关键词和 grep。

## 5. 语雀配置

```json
{
  "yuque_api_base": "https://nova.yuque.com/api/v2",
  "yuque_space_login": "qc19gt",
  "approved_repositories": [],
  "excluded_repositories": [
    {
      "__template_key": "repository_exclusion",
      "namespace": "qc19gt/ogaye8",
      "reason": "QA 成品仓库，默认不作为缺口调查源"
    }
  ],
  "purge_excluded_repository_data": true
}
```

`qc19gt/ogaye8` 只是默认排除项，不是写死规则。取消排除后仓库也不会立即下载，仍需进入批准列表。保存配置并重载插件后，使用 `/nju_collect repo sync` 下载批准仓库正文并建立本地索引；`/nju_collect repo status` 可查看结果。

## 6. 邮件与调度配置

首次全流程测试完成前保持关闭：

```json
{
  "timezone": "Asia/Shanghai",
  "daily_report_time": "00:00",
  "daily_report_enabled": false,
  "smtp_host": "",
  "smtp_port": 465,
  "smtp_username": "",
  "smtp_password": "",
  "smtp_use_ssl": true,
  "mail_from": "",
  "mail_recipients": [],
  "mail_subject_prefix": "NJU 知识库日报",
  "attach_full_html": true
}
```

先用 `/nju_collect test startup live` 验证 SMTP 登录；该测试不会发信。手动生成日报后，
用 `/nju_collect report preview <日期>` 检查 HTML，再用 `/nju_collect report send <日期>`
发送。成功投递的“报告版本 + 收件人”不会重复发送；失败项可以再次执行相同命令重试。

## 7. 当前可执行的上线前自检

安装到 AstrBot 后，运维管理员可在私聊或群聊运行：

```text
/nju_collect test startup
/nju_collect test startup live
```

第一条只检查本地数据库、后台写入、目标群、模型配置、语雀/邮件配置和导出目录。
加上 `live` 后会真实连接 LLM、语雀和 SMTP，但不会下载语雀仓库正文，也不会发送邮件。
检查输出不会显示 Token、密码或完整外部错误内容。

## 8. 推荐启用顺序

1. 安装插件，保持消息采集和日报关闭；
2. 填写目标群、群别名和两类 QQ 权限；
3. 开启消息采集，通过 `/nju_collect status` 检查是否落库；
4. 配置 LLM Provider，测试问题范围判断和 AI 自动审核；
5. 填写语雀 Token 和允许调查的仓库，运行 `/nju_collect repo sync`；
6. 配置 SMTP，运行一次历史或指定日期完整处理；
7. 预览 HTML，显式发送第一封日报并确认收件正常；
8. 把 `daily_report_time` 设为希望处理前一日数据的时间，最后开启每日日报任务。

长任务采用管理员主动查询进度，不向聊天连续推送：语雀同步使用
`/nju_collect repo status`，日报处理使用 `/nju_collect report status`。普通运行跳过完整日期；
强制重跑必须使用 `/nju_collect report rerun YYYY-MM-DD confirm`。

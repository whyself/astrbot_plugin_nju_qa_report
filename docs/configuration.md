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
- `llm_provider_id`：从 AstrBot 已配置的对话模型中选择，留空时使用当前默认模型；
- `embedding_provider_id`：后续向量检索所用的 Embedding Provider ID。

不要把以上值写入 `.env` 后提交，也不要在 GitHub Issue 中粘贴配置文件。

## 3. 最小消息采集配置

首次安装默认关闭采集。先填写：

```json
{
  "capture_mode": "selected_groups",
  "target_group_ids": ["你的群号"],
  "group_aliases": {
    "你的群号": "迎新群"
  },
  "capture_queue_size": 5000,
  "raw_message_retention_days": 90,
  "report_viewer_qq_ids": ["可查看日报的QQ号"],
  "operator_qq_ids": ["运维管理员QQ号"],
  "capture_enabled": true
}
```

当前仓库已把目标群 `826811581` 及别名“南京大学迎新群”作为默认值预填，
但 `capture_enabled` 仍默认关闭，因此安装后不会在未确认配置的情况下自动采集。

消息监听器只做快速入队，后台单写入线程负责 SQLite 落库，因此数据库锁不会阻塞 AstrBot 的其他消息插件。若队列异常溢出，`/nju_collect status` 会显示积压和丢弃数量。超过保留期的原始消息会在插件启动时自动清理。

如果查看人员或运维人员本身已在 AstrBot 全局管理员列表中，可以保留下面两个默认配置：

```json
{
  "inherit_astrbot_admins_as_viewers": true,
  "inherit_astrbot_admins_as_operators": true,
  "sensitive_commands_private_only": true
}
```

查看人员与运维管理员是两种独立权限。需要同时拥有两种权限时，应同时加入两个列表，或通过两个 AstrBot 管理员继承开关获得权限。

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

消息到达群聊时只写入本地数据库，不实时调用模型。AI 初筛和自动复核在每日批处理阶段运行。技术错误会自动重试，不会被伪装成“低质量问题”；语义上连续无法确认的问题才会自动低置信排除。

## 5. 语雀配置

```json
{
  "yuque_api_base": "https://nova.yuque.com/api/v2",
  "yuque_space_login": "qc19gt",
  "approved_repositories": [],
  "excluded_repositories": [
    {
      "namespace": "qc19gt/ogaye8",
      "reason": "QA 成品仓库，默认不作为缺口调查源"
    }
  ],
  "purge_excluded_repository_data": true
}
```

`qc19gt/ogaye8` 只是默认排除项，不是写死规则。取消排除后仓库也不会立即下载，仍需进入批准列表。语雀同步功能完成前，可以暂时不填 Token 和批准仓库。

## 6. 邮件与调度配置

日报功能完成前保持关闭：

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
  "mail_recipients": []
}
```

后续完成邮件模块后，先使用管理员私聊测试指令验证 SMTP，再开启 `daily_report_enabled`。

## 7. 当前可执行的上线前自检

安装到 AstrBot 后，运维管理员可私聊运行：

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
3. 开启消息采集，通过私聊 `/nju_collect status` 检查是否落库；
4. 配置 LLM Provider，测试问题范围判断和 AI 自动审核；
5. 填写语雀 Token，扫描仓库元数据，再批准允许调查的仓库；
6. 配置 SMTP 并发送测试邮件；
7. 最后开启每日日报任务。

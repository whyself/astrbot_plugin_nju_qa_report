# QQ Chat Exporter 历史记录导入

本插件支持 [QQ Chat Exporter](https://github.com/shuakami/qq-chat-exporter) 当前格式的：

- 普通单文件 JSON；
- 大群分块 JSONL ZIP（包含 `manifest.json` 和 `chunks/*.jsonl`）。

不建议用 HTML、TXT 或 Excel：这些格式缺少稳定消息 ID、结构化发送者和回复关系，
不能保证幂等导入和回答关联。

## 1. 安装并启动 QQ Chat Exporter

Windows 普通用户：

1. 打开 [QQ Chat Exporter Releases](https://github.com/shuakami/qq-chat-exporter/releases)。
2. 下载 `NapCat-QCE-Windows-x64-vxxx.zip`，不要下载 Source code。
3. 完整解压后运行 `launcher-user.bat`。
4. 用手机 QQ 扫码登录能够查看目标群历史记录的 QQ 账号。
5. 控制台出现 Web 地址和 Token 后，打开 `http://localhost:40653/qce`。
6. 输入控制台显示的 Access Token。

Access Token 只用于打开本机 QCE 页面，不要上传到 AstrBot、GitHub 或发给其他人。

## 2. 导出目标群

1. 在 QCE 左侧进入“会话（Sessions）”。
2. 找到群号 `826811581` 对应的“南京大学迎新群”。
3. 点击“导出”。
4. 格式选择 **JSON**。
5. 时间范围留空表示导出全部历史；首次测试也可以先选择最近 7～30 天。
6. 图片、视频等媒体资源不是本插件分析所需，可选择快速导出/跳过资源下载。
7. 创建任务，到“任务”页面等待完成。

普通规模直接导出单文件 JSON。若记录非常多，在高级选项中启用分块/流式 JSONL，
并将包含 `manifest.json` 与 `chunks` 目录的完整结果打成 ZIP；不要只上传某一个 chunk。

## 3. 上传到 AstrBot

1. 更新并重载 `astrbot_plugin_nju_qa_report`。
2. 打开插件配置。
3. 在“QQ Chat Exporter 历史记录文件”上传 `.json` 或 `.zip`。
4. “历史记录中需要排除的机器人 QQ 号”只填写当时确实自动发言的机器人账号。
5. 保存并重载插件。

导出文件包含真实聊天内容，只应上传到你自己控制的 AstrBot。不要提交到 GitHub。

## 4. 检查并导入

运维管理员私聊机器人运行：

```text
/nju_collect import inspect
```

插件会检查：

- 文件是否为 QCE 群聊 JSON；
- 群号是否匹配 `target_group_ids` 中的 `826811581`；
- 消息数量和时间范围；
- 分块 ZIP 是否包含全部 manifest/chunk 文件。

确认输出正确后运行：

```text
/nju_collect import run
```

导入会自动：

- 跳过系统消息、已撤回消息、Bot 消息和 `/` 开头的指令；
- 保留文本、附件占位符和原生回复关系；
- 以 QCE 消息 ID 去重；
- 每 1000 条批量写入 SQLite；
- 拒绝其他群的导出文件。

同一文件可以重复执行，已导入消息会计入“重复”，不会产生第二份记录。

## 5. 处理历史日期

导入完成后运行：

```text
/nju_collect status
/nju_collect report run all
```

再按日期查看：

```text
/nju_collect report status 2026-07-01
/南哪日报 列表 2026-07-01
/南哪日报 导出
```

确认导入成功后，可以从插件配置移除上传文件并保存。已经写入 SQLite 的消息不会因此删除，
仍按 `raw_message_retention_days` 配置清理。

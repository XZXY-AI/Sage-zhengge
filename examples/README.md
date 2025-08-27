# SAgent CLI 使用指南

## 简介

SAgent CLI 是一个基于命令行的智能代理交互工具，支持流式对话、工具调用、多智能体协作等功能。通过美观的消息框架显示不同类型的消息，提供良好的用户体验。

## 功能特性

- 🤖 **智能对话**：支持与AI智能体进行自然语言对话
- 🔧 **工具集成**：集成MCP工具，支持文件操作、网络搜索等功能
- 🧠 **深度思考**：可选启用深度思考模式，提供更详细的推理过程
- 👥 **多智能体**：支持多智能体协作，处理复杂任务
- 🎨 **美观界面**：彩色消息框架，不同消息类型有不同的视觉效果
- ⚡ **流式输出**：实时显示AI响应，提供流畅的交互体验

## 安装要求

确保已安装以下依赖：

```bash
pip install rich openai asyncio
```

## 配置文件

### 1. MCP设置文件 (mcp_setting.json)

配置MCP工具服务器：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["@modelcontextprotocol/server-filesystem", "/path/to/workspace"],
      "env": {}
    }
  }
}
```

### 2. 预设运行配置 (preset_running_config.json)

该json配置，可以在agent development的页面中导出。

配置默认参数：

```json
{
  "llmConfig": {
    "model": "deepseek/deepseek-chat",
    "maxTokens": 4096,
    "temperature": 0.2
  },
  "deepThinking": false,
  "multiAgent": false,
  "availableTools": [],
  "systemPrefix": "你是一个有用的AI助手。",
  "maxLoopCount": 10,
  "available_workflows": {},
  "system_context": {}
}
```

## 使用方法

### 基本用法

```bash
python sagents_cli.py --api_key YOUR_API_KEY --model deepseek/deepseek-chat --base_url https://api.deepseek.com
```

### 完整参数示例

```bash
python sagents_cli.py \
  --api_key YOUR_API_KEY \
  --model deepseek/deepseek-chat \
  --base_url https://api.deepseek.com \
  --max_tokens 4096 \
  --temperature 0.2 \
  --workspace ./workspace \
  --mcp_setting_path ./mcp_setting.json \
  --preset_running_config_path ./preset_running_config.json
```

### 高级选项

```bash
# 启用深度思考模式
python sagents_cli.py --api_key KEY --model MODEL --base_url URL

# 禁用深度思考
python sagents_cli.py --api_key KEY --model MODEL --base_url URL --no-deepthink

# 禁用多智能体
python sagents_cli.py --api_key KEY --model MODEL --base_url URL --no-multi-agent

# 指定工具目录
python sagents_cli.py --api_key KEY --model MODEL --base_url URL --tools_folders ./tools ./custom_tools
```

## 命令行参数

| 参数                             | 必需 | 说明             | 默认值                       |
| -------------------------------- | ---- | ---------------- | ---------------------------- |
| `--api_key`                    | ✅   | API密钥          | -                            |
| `--model`                      | ✅   | 模型名称         | -                            |
| `--base_url`                   | ✅   | API基础URL       | -                            |
| `--tools_folders`              | ❌   | 工具目录路径     | []                           |
| `--max_tokens`                 | ❌   | 最大令牌数       | 4096                         |
| `--temperature`                | ❌   | 温度参数         | 0.2                          |
| `--no-deepthink`               | ❌   | 禁用深度思考     | False                        |
| `--no-multi-agent`             | ❌   | 禁用多智能体     | False                        |
| `--workspace`                  | ❌   | 工作目录         | ./agent_workspace            |
| `--mcp_setting_path`           | ❌   | MCP设置文件路径  | ./mcp_setting.json           |
| `--preset_running_config_path` | ❌   | 预设配置文件路径 | ./preset_running_config.json |

## 交互界面

### 消息类型

程序会显示不同类型的消息，每种类型都有独特的颜色和图标：

- 🔧 **工具调用** (黄色)
- ⚙️ **工具结果** (黄色)
- 🎯 **子任务结果** (红色)
- ❌ **错误** (红色)
- ⚙️ **系统** (黑色)
- 💬 **普通消息** (蓝色)

### 操作命令

在对话中可以使用以下命令：

- `exit` 或 `quit`：退出程序
- `Ctrl+C`：强制退出

## 示例对话

```
欢迎使用 SAgent CLI。输入 'exit' 或 'quit' 退出。
你: 帮我创建一个Python文件
SAgent:

╭─────────────────────────────────────────╮
│ 🔧 工具调用                             │
├─────────────────────────────────────────┤
│ file_write(content="print('Hello')",    │
│ file_path="hello.py")                   │
╰─────────────────────────────────────────╯

╭─────────────────────────────────────────╮
│ ⚙️ 工具结果                               │
├─────────────────────────────────────────┤
│ 文件已成功创建：hello.py                   │
╰─────────────────────────────────────────╯
```

## 故障排除

### 常见问题

1. **API密钥错误**

   - 检查API密钥是否正确
   - 确认API服务可用
2. **模型不存在**

   - 验证模型名称是否正确
   - 检查API提供商支持的模型列表
3. **工具调用失败**

   - 检查MCP设置文件配置
   - 确认工具服务器正常运行
4. **配置文件错误**

   - 验证JSON格式是否正确
   - 检查文件路径是否存在

### 调试模式

如果遇到问题，可以查看详细的错误信息：

```bash
# 程序会自动显示错误堆栈信息
# 检查日志输出以获取更多调试信息
```

## 开发说明

### 代码结构

- `sagents_cli.py`：主程序入口
- `mcp_setting.json`：MCP工具配置
- `preset_running_config.json`：预设运行配置

### 自定义配置

可以通过修改配置文件来自定义：

- 默认模型参数
- 系统提示词
- 可用工具列表
- 工作流配置

## 更新日志

- 支持流式消息框显示
- 优化emoji字符宽度计算
- 修复边框对齐问题
- 改进颜色标记处理

## 许可证

本项目遵循相应的开源许可证。

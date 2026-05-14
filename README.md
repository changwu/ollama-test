# Ollama + Gemma（社科研究提示）Python 调用示例

## 前置条件

- 已安装 Ollama，且本机服务可访问：`http://localhost:11434`
- 本地已拉取任意一个 gemma 模型（名称以 `gemma` 开头）
  - 例如：`ollama pull gemma2:2b`
- 已安装 Python 3.10+（建议 3.12）。若刚安装完 Python，需关闭并重新打开终端以刷新 PATH

## 运行

```powershell
python .\run_soc_sci_demo.py --task open_coding
python .\run_soc_sci_demo.py --task sentiment5
python .\run_soc_sci_demo.py --task all
```

如果你使用的是带“思考/推理”输出的模型（例如部分 gemma4 变体），且发现结果为空，通常是因为生成长度被 `max_tokens` 截断。可在运行时提高上限：

```powershell
python .\run_soc_sci_demo.py --task all --max-tokens 1500
```

可选：创建虚拟环境（本项目无第三方依赖，但建议隔离环境）

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python .\run_soc_sci_demo.py --task all
```

指定模型（当你本地有多个 gemma 变体时）：

```powershell
python .\run_soc_sci_demo.py --model gemma2:2b --task all
```

指定 Ollama 地址（非默认端口或远端）：

```powershell
python .\run_soc_sci_demo.py --base-url http://127.0.0.1:11434 --task all
```

## 文件说明

- `ollama_http.py`：纯标准库实现的 Ollama HTTP 客户端（`/api/tags`、`/api/chat`）
- `soc_sci_tasks.py`：你的社科研究任务 prompt 模板与示例输入
- `run_soc_sci_demo.py`：命令行 demo，选择任务并打印模型输出

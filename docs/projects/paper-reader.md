# Paper Reader Agent

**类型**：开源项目 | **GitHub**：[GoDiao/Paper-Reader](https://github.com/GoDiao/Paper-Reader)

分层多智能体系统，用于学术论文深度分析和AI驱动的网络研究。

![Paper Reader界面](../images/webui_en.png)

## 核心功能

**分层多智能体架构（1+3+1）** - 模拟专业研究团队

* **Architect**：解构论文并规划阅读策略
* **Specialist Team**：Context、Math、Data三个并行专家
* **Editor**：合成出版级报告并嵌入图表

**视觉理解** - 检测、提取并"看见"图表，直接嵌入分析中

**Deep Research** - 基于Tavily和Valyu的AI网络研究工具

**双语报告** - 同时生成英文和中文原生质量报告

![系统架构](../images/archv1.png)

## 技术架构

**后端**：FastAPI + 异步编排 + 多智能体系统

**前端**：现代Web UI + 实时进度追踪

**PDF解析**：PyMuPDF（快速）/ MinerU（高保真）双引擎

**LLM**：DeepSeek / OpenAI兼容API

![报告预览](../images/report_preview.png)

## 关键特性

* 智能PDF解析管道（表格提取、数学公式区域检测）
* 实时流式响应（逐token生成报告）
* 迭代分析与Gap Agent（自动识别信息缺口）
* 一键导出到Notion（原生表格、LaTeX公式）
* 研究历史管理（浏览、搜索、删除）
* 多种引用格式（Numbered、APA、MLA、Chicago）

## 快速开始

```bash
git clone https://github.com/GoDiao/Paper-Reader.git
cd Paper-Reader
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python web_server.py
```

访问 `http://localhost:8000` 使用Paper Reader，或访问 `http://localhost:8000/researcher` 使用Deep Research。

## 项目链接

* [GitHub仓库](https://github.com/GoDiao/Paper-Reader)
* [返回项目列表](./README.md)

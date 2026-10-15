# Bert-Lab：BERT 标点恢复与批量长文推理

《大模型实践与应用》课程实验：微调中文 BERT，预测每个字符后面的标点；使用 **400 字窗口 + 300 字唯一归属区 + Batch 推理** 处理长文本。

📄 [实验报告](新实验报告.md) · [完整运行说明](docs/复现说明.md) · [课程网站图片引用](docs/课程网站提交说明.md)

## 实验结果

- 标点 micro-F1：1 月参考测试 **78.79%**，2 月冻结测试 **78.72%**，《论语》参考 **51.45%**。
- 同一权重下，12 篇自然长文批量推理较逐窗基线快 **4.40 倍**；十万字拼接压力输入快 **4.60 倍**。
- 四组各三轮训练、三个随机种子、加权对照、33 个性能配置各七次计时。

以上是 RTX 5060 Laptop 上的实测结果；现代新闻效果不能代表所有古文，完整限制与失败案例见报告。

## 安装

Python 3.12，支持 BF16 的 NVIDIA GPU。本实验使用 PyTorch 2.7.1 / CUDA 12.8。

```powershell
git clone git@github.com:70clementine/Bert-Lab.git
cd Bert-Lab
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
```

## 使用

训练完成后，默认模型位于 `runs/v2_corrected_seed42/best`：

```powershell
.\.venv\Scripts\python.exe bert_punct.py predict --text "学而时习之不亦说乎有朋自远方来不亦乐乎" --method fast --batch-size 8
.\.venv\Scripts\python.exe bert_punct.py predict --input input.txt --output output.txt --method fast --batch-size 8
```

输入文件为 UTF-8。模型恢复 `，。？！；：、`，不恢复完整排版或引号嵌套。`fast` 为批量实现，`fixed` 为参考实现，`sequential` 为逐窗基线，`fusion` 为重叠融合。

## 仓库内容

- `bert_punct.py`、`revision_*.py`：数据准备、微调、评测、速度测试与独立复算。
- `configs/`：四组训练配置；`test_correctness.py`、`verify_gpu_edges.py`：编码、窗口及 GPU 测试。
- `results/`：已完成实验的指标、训练日志摘要、原始计时与数据清单，供核查报告。
- `artifacts/`：报告图片；`tools/export_course_report.py`：生成使用网络图片地址的报告副本。

**仓库不含人民日报语料、完整处理数据、逐字预测、预训练/微调大权重或虚拟环境。** 原始模型由下载脚本获取；微调权重需按[运行说明](docs/复现说明.md)训练生成。`results/` 是已完成实验的记录，不是可直接加载的模型。源数据与预训练模型按其各自使用条件获取。

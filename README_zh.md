# 多环境应力耦合失效机理可视化仿真系统

English version: [README.md](README.md)

本仓库提供一个基于 Streamlit 的可视化仿真程序，用于研究多环境应力耦合作用下的剩余使用寿命（RUL）预测与失效机理分析。

实验以 NASA C-MAPSS FD004 涡扇发动机退化数据集为基础，主要集成了以下模块：

- 基于 LightGBM 的 RUL 预测
- Weibull 可靠性建模
- 隐马尔可夫模型（HMM）退化状态演化分析
- 基于 SHAP 的模型可解释性分析
- 蒙特卡洛寿命采样仿真

## 仓库内容

本仓库是课程项目提交版本，包含以下主要内容：

- 可视化仿真程序源码
- 课程结课论文及其 LaTeX 源码
- 课堂汇报幻灯片及讲稿
- 论文与汇报使用的图表资源

## 环境要求

- Python 3.10 及以上
- Windows、macOS 或 Linux
- 无需 GPU

安装依赖：

```bash
pip install -r requirements.txt
```

在高核数服务器上，建议限制数值库线程数，避免 OpenBLAS 或 LightGBM 过度并行：

```bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LGBM_NUM_THREADS=4
```

## 数据说明

NASA C-MAPSS FD004 数据文件位于 `data/` 目录下：

```text
data/
├── train_FD004.txt
├── test_FD004.txt
└── RUL_FD004.txt
```

当前课程提交工作区已经包含这些数据文件。

## 运行方式

启动可视化仿真系统：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LGBM_NUM_THREADS=4 streamlit run app.py
```

若需远程访问，可使用：

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LGBM_NUM_THREADS=4 \
streamlit run app.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

首次访问会自动加载数据并训练模型，后续交互会复用缓存结果。

## 论文与幻灯片编译

编译课程论文：

```bash
cd paper
xelatex -interaction=nonstopmode -halt-on-error paper.tex
```

编译汇报幻灯片：

```bash
cd slide
xelatex -interaction=nonstopmode -halt-on-error presentation.tex
```

## 系统功能

1. **Weibull 可靠性仿真**  
   比较不同应力类型、不同应力水平下的可靠度曲线。

2. **退化状态演化分析**  
   展示 HMM 状态路径和前向状态概率预测结果。

3. **RUL 预测对比**  
   对比单应力模型与多应力耦合模型的寿命预测表现。

4. **蒙特卡洛寿命仿真**  
   调节应力倍率，观察寿命分布和经验可靠度曲线变化。

5. **模型评估指标汇总**  
   展示 RMSE、MAE、R²、±20 周期准确率和 PHM08 分数。

6. **SHAP 模型解释**  
   展示全局特征重要性、依赖图和局部解释结果。

## 项目结构

```text
course_simulation/
├── app.py              # Streamlit 应用入口
├── requirements.txt    # Python 依赖
├── README.md           # 英文说明
├── README_zh.md        # 中文说明
├── SUBMISSION_CHECKLIST.md
├── data/
│   ├── README.md
│   ├── train_FD004.txt
│   ├── test_FD004.txt
│   └── RUL_FD004.txt
├── paper/              # 课程论文源码与 PDF
├── slide/              # 幻灯片源码、PDF 与讲稿
├── scripts/            # 图表生成脚本
├── src/
│   ├── __init__.py
│   └── engine.py       # 仿真与建模主流程
└── results/            # 运行输出目录
```

## 许可证

本项目采用 MIT License。

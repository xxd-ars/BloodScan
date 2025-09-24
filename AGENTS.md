# Repository Guidelines

## 项目结构与模块组织
- `src/` 存放核心代码：`main.py` 调度双光源检测流程，`algorithm/` 聚合特征提取与融合算子，`ui/` 提供 WinUI 界面，`motor/` 管理传动控制，`utils/` 整理公用工具。
- `scripts/` 专注数据与推理辅助脚本（如 `annotate_and_crop.py`、`yolo_service.py`），用于标注裁剪、远程验证与可视化。
- `ultralytics/` 为外部 YOLO 依赖，默认只读；如需自定义，请在 `src/` 内编写包装层避免直接改动。
- 数据目录（`data/`、`datasets/`、`runs/` 等）不纳入版本库，若影响实验结果，请在 PR 说明中标注本地路径或权重 ID。

## 构建、测试与开发命令
- `python -m venv .venv; .venv\\Scripts\\activate`：在 Windows 上创建并激活虚拟环境。
- `pip install -r requirements.txt`：安装推理、训练与 UI 所需依赖。
- `python src/main.py`：从采集到检测输出的端到端运行流程。
- `python -m pytest tests`：执行自动化测试，可追加 `-k case_name` 精选用例。
- `python scripts/test_visualization.py --weights dual_yolo/...`：对新权重进行快速可视化冒烟验证。

## 代码风格与命名规范
- 遵循 PEP 8：四空格缩进，函数/模块使用 `snake_case`，类名使用 `CamelCase`，目录保持小写。
- 关键接口尽量补充类型注解，尤其是 `algorithm` 与 `ui`、`motor` 之间的数据约定，便于跨团队协作。
- 实验输出建议命名为 `runs/<日期>_<摘要>`，便于回溯与对比。
- 公共函数提供简洁 docstring；涉及传感器或电机单位时在行内注明，避免歧义。

## 测试指南
- 按源码结构在 `tests/` 下建立对应测试文件（示例：`tests/yolo_seg/test_decoder.py` 对应 `src/algorithm/decoder.py`）。
- 优先使用 `pytest` fixture 构造样例帧与配置，避免硬编码绝对路径。
- 提交前运行 `python -m pytest tests` 并补充相关脚本冒烟测试。
- 若模型或流程变更影响指标，请在 PR 中记录 mAP、推理延迟等关键数值。

## 提交与 Pull Request 规范
- 沿用当前日志风格：简短祈使句，可按需添加作用域前缀（如 `yolo: 调整置信度`）。
- 在提交正文中注明数据集、权重或实验 ID，方便复现背景。
- PR 需概述目标、列出验证命令，并附上 UI 截图或结果图（若界面或指标发生变化）。
- 按子系统（视觉、UI、硬件等）指派审阅者，并关联驱动本次改动的 issue 或研究记录。
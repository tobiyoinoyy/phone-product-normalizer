# Phone Product Normalizer

只使用 **ZhengPeng7/BiRefNet_dynamic** 处理手机产品实拍图：抠图 → 保守去支架 → 自动摆正 → 紧裁切透明 PNG。保留原图细节，不重绘瑕疵、镜头、标签或反光。

## 安装与使用

需要 Python 3.9+。在项目目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m phone_normalizer photos/ --output-dir outputs/结果
```

模型首次运行时自动下载，后续复用本地缓存。支持单图、多图和文件夹；输出 PNG 与处理记录 JSON，不覆盖输入。没有模型切换参数。

macOS 用户可双击 **手机图片一键处理.command**，按提示安装依赖后，在窗口中选择图片或文件夹。详见 [设计师使用说明](设计师使用说明.md)。Python 集成见 [开发交付说明](开发交付说明.md)。

## 处理约定

- 推理保持画幅比例，默认最长边 1024 px，补齐到 32 的倍数；alpha 恢复到原图尺寸。
- 去支架证据不足时保留原样；平面摆正不代替透视校正。
- 旋转使用预乘 alpha，减少透明边缘的白边和黑边。
- 每张图后清理推理缓存；MPS 内存不足时用同一个模型在 CPU 重试。
- 几何阶段默认最多处理 6400 万像素源图。

运行 `python -m phone_normalizer --help` 查看设备、裁切、支架及内存参数。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `phone_normalizer/` | Python API 和命令行 |
| `scripts/segmentation.py` | 唯一模型的推理与处理流程 |
| `scripts/geometry_postprocess.py` | 去支架、摆正、透明边缘裁切 |
| `scripts/batch_normalize.py` | 桌面窗口的批处理与进度回调 |
| `scripts/phone_normalizer_gui.py` | 处理窗口 |
| `tests/` | 无需下载权重的回归测试 |
| `tools/build_delivery.py` | 开发版、macOS 设计师版打包 |
| `SKILL.md` | Codex 技能说明 |

## 验证与交付

```bash
.venv/bin/python tests/test_segmentation.py
.venv/bin/python tests/test_geometry_postprocess.py
.venv/bin/python tests/test_delivery_api.py
.venv/bin/python tests/test_batch_normalize.py
.venv/bin/python tests/test_macos_entry.py
.venv/bin/python tools/build_delivery.py
```

交付包见项目旁的 `交付包/`。Git 仅管理源码、测试和使用说明；原始照片、输出图片、模型权重与虚拟环境不上传。

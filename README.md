# Phone Product Normalizer

只使用 **ZhengPeng7/BiRefNet_dynamic** 处理手机产品实拍图：抠图、保守去支架、自动摆正，输出透明 PNG。v0.3 新增六面精裁与可选的 UV 模板对齐入口，用于减少裁切边界与主体位置变化造成的贴图偏移。

处理使用原图内容，不生成或涂抹瑕疵、镜头、标签、接口和反光；旋转、缩放等几何变换会通过插值重采样。

## 安装与使用

需要 Python 3.9+。在项目目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

模型首次运行时自动下载，后续复用本地缓存。没有模型切换参数。

### 六面精裁

准备一个六面文件夹，包含 `front.jpg`、`back.jpg`、`left.jpg`、`right.jpg`、`top.jpg`、`bottom.jpg`，每面一张。文件名大小写不敏感，支持 JPG、PNG。

```bash
.venv/bin/python -m phone_normalizer.uv_cli photos/sku1 --output-dir outputs/sku1
```

该入口默认只做精裁，以 alpha 阈值 128 确定裁切边界，保留裁切框内的连续透明度。六面朝向沿用当前拍摄约定：**left 逆时针 90°，right 顺时针 90°**，其余四面不做 90° 旋转或镜像。输入应为符合此约定的原始六面照片。

只做精裁时，各张图的输出尺寸仍由各自的主体边界决定。要让同一套 UV 连续替换图片，应使用对应的 UV 模板。

### 按认可的 UV 模板对齐

```bash
.venv/bin/python -m phone_normalizer.uv_cli photos/sku1 \
  --output-dir outputs/sku1-uv \
  --uv-profile iphone12pro-user-crop-v1
```

此模板复现本轮用户认可的六面裁边与坐标约定，固定输出以下尺寸（宽 × 高）：

| front | back | left | right | top | bottom |
| --- | --- | --- | --- | --- | --- |
| 957 × 1985 | 960 × 1986 | 104 × 1904 | 116 × 1877 | 1017 × 130 | 1046 × 112 |

模板对齐先把主体边界放到参考坐标，再应用认可的裁切区域；必要时只做受限的全局范围校正。它不只是把图片拉到相同宽高，也不代表镜头、按钮等所有局部都能逐像素对齐。**该模板只对应本次拍摄方向与 UV 布局，不是所有 iPhone 12 Pro 的通用模板。** 使用前后均应在目标模型上检查贴合效果。

自定义数字模板可通过 `--uv-profile path/to/profile.json` 指定。模板只记录数值、朝向和裁切约定，不需要打包参考照片或私人路径。详细流程、适用边界和自定义说明见 [UV 模板说明](docs/uv-profile.md)。

### 原有命令与桌面窗口

原有 CLI、Python API 和桌面窗口继续兼容原流程。单图、多图或普通文件夹处理仍可使用：

```bash
.venv/bin/python -m phone_normalizer photos/ --output-dir outputs/结果
```

不要将新 UV 入口的默认精裁设置误认为旧入口或 GUI 的设置已经改变。输出写入新目录，保留原图。

macOS 用户可双击 **手机图片一键处理.command**，按提示安装依赖后，在窗口中选择图片或文件夹。详见 [设计师使用说明](设计师使用说明.md)。Python 集成见 [开发交付说明](开发交付说明.md)。

## 处理约定

- 推理保持画幅比例，默认最长边 1024 px，补齐到 32 的倍数；alpha 恢复到原图尺寸。
- 去支架证据不足时保留原样；平面摆正不代替透视校正。
- 旋转使用预乘 alpha，减少透明边缘的白边和黑边。
- 精裁只解决裁切边界问题；固定 UV 坐标需要另选合适的模板。镜头凸起和圆角造成的局部透明区域不等于多余留白。
- 每张图后清理推理缓存；MPS 内存不足时用同一个模型在 CPU 重试。
- 几何阶段默认最多处理 6400 万像素源图。

运行 `python -m phone_normalizer.uv_cli --help` 查看六面入口参数；原入口参数见 `python -m phone_normalizer --help`。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `phone_normalizer/` | Python API 和命令行 |
| `phone_normalizer/uv_cli.py` | 六面精裁与可选 UV 模板入口 |
| `scripts/segmentation.py` | 唯一模型的推理与处理流程 |
| `scripts/geometry_postprocess.py` | 去支架、摆正、透明边缘裁切 |
| `scripts/batch_normalize.py` | 桌面窗口的批处理与进度回调 |
| `scripts/phone_normalizer_gui.py` | 处理窗口 |
| `tests/` | 无需下载权重的回归测试 |
| `tools/build_delivery.py` | 开发版、macOS 设计师版打包 |
| `SKILL.md` | Codex 技能说明 |
| `docs/uv-profile.md` | UV 模板约定与验证边界 |

## 验证与交付

```bash
.venv/bin/python tests/test_segmentation.py
.venv/bin/python tests/test_geometry_postprocess.py
.venv/bin/python tests/test_precise_geometry.py
.venv/bin/python tests/test_uv_profile.py
.venv/bin/python tests/test_uv_pipeline.py
.venv/bin/python tests/test_delivery_api.py
.venv/bin/python tests/test_batch_normalize.py
.venv/bin/python tests/test_macos_entry.py
.venv/bin/python tools/build_delivery.py
```

v0.3 发布回归：两套共 12 张本地样图使用此前缓存的 BiRefNet_dynamic 分割结果，经过正式精裁、定向、模板对齐流程后，与用户认可输出的 RGBA 逐像素一致（12/12）。这验证了处理逻辑迁移的复现性；没有增加新的跨 SKU 样本，也不替代目标模型中的 UV 检查。安装包已验证包含数值模板并可独立导入。

交付包见项目旁的 `交付包/`。Git 仅管理源码、测试、使用说明和不含图片的数字模板；原始照片、输出图片、私人路径、模型权重与虚拟环境不上传。

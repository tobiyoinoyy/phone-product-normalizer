# Phone Product Normalizer：交付版入口

本目录保留旧的 `scripts/normalize_phone.py` 和
`scripts/normalize_ben2.py` 命令，同时新增可安装的 `phone_normalizer` 包。
新包面向批量交付，默认使用 `ZhengPeng7/BiRefNet_dynamic`；旧脚本的默认行为不变。

完整的开发/设计师交接清单见 [`开发交付说明.md`](开发交付说明.md)。

## 安装

在本目录执行（推荐使用独立虚拟环境）：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e ".[birefnet]"
```

也可以继续使用现有的 `requirements-birefnet.txt`。模型权重首次运行时从
Hugging Face 下载并缓存。

## 命令行

传入一张或多张图片，也可以传入文件夹。文件夹默认递归扫描，输出目录中每张图片会得到
一个透明 PNG 和一个同名 JSON 记录：

```bash
python3 -m phone_normalizer photo.jpg another-photo.jpg \
  --output-dir outputs/结果 \
  --manifest outputs/结果/manifest.json
```

新入口默认等价于：

```text
backend = birefnet
model = ZhengPeng7/BiRefNet_dynamic
birefnet-resolution = auto
device = auto
```

动态 BiRefNet 默认启用内存保护：模型推理最长边限制为 `1024 px`，保持比例并补齐到 32
的倍数；每张图片结束后清理临时缓存。MPS 内存不足时会自动切换到 CPU，并以最长边
`1024 px` 重试。需要更保守时可传 `--birefnet-max-side 768`；`none` 会关闭上限，可能
让高分辨率输入再次占满统一内存，因此只建议在受控环境使用。

去支架、摆正和紧裁切使用源图像分辨率，默认另有 `64 MP`（`64,000,000` 像素）几何护栏。
超过该上限的单张图片会在几何阶段开始前明确失败并继续批处理其他图片，不会分配可能导致
内存耗尽的整幅临时数组。可显式传 `--max-geometry-pixels none` 关闭，但不建议日常使用。

该最长边保护只作用于 dynamic 模型的自动尺寸路径。显式指定
`--birefnet-resolution`，或使用 HR/lite-2K 固定分辨率权重时，首次推理仍会按指定/推荐尺寸
运行；需要更低内存时请改用较小的固定分辨率或 dynamic 默认路径。CPU 回退上限用于加速器
首次尝试失败后的重试；届时固定分辨率可能按该上限缩小以保护机器，实际模型输入尺寸会记录
在逐图 JSON 中。

macOS 启动器和底层脚本还会设置额外的 MPS 分配器护栏（默认
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`、`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6`、
`PYTORCH_ENABLE_MPS_FALLBACK=1`），并尊重调用环境中已经设置的值。它们是辅助保护，不是对
峰值内存的硬性保证。

常用切换：

```bash
# BEN2 对照（旧 normalize_ben2.py 的默认入口仍保留）
python3 -m phone_normalizer photos --backend ben2 --model PramaLLC/BEN2

# BiRefNet 标准权重
python3 -m phone_normalizer photos --backend birefnet --model ZhengPeng7/BiRefNet

# 只输出 PNG，不写逐图 JSON
python3 -m phone_normalizer photos --no-metadata
```

未知后端会明确报错，不会静默换模型。`--no-support-removal`、`--no-deskew`、
`--no-tight-crop` 可分别关闭几何阶段；`--confidence-floor` 可调整支架自动裁切的
置信度门槛；`--require-support-border` 只接受延伸到画面边界的支架候选。还可使用
`--birefnet-cpu-fallback-max-side`、`--max-geometry-pixels`、`--no-memory-cleanup` 和
`--no-cpu-fallback` 调整内存策略（默认开启保护，不建议关闭）。

## Python API

```python
from phone_normalizer import Normalizer, NormalizerConfig

config = NormalizerConfig()  # BiRefNet_dynamic + auto resolution
normalizer = Normalizer(config)
results = normalizer.process_many(
    ["photos/"],
    output_dir="outputs/结果",
    save_metadata=True,
    manifest_path="outputs/结果/manifest.json",
)
```

同一个 `Normalizer` 实例会复用已加载的分割后端，适合批处理。应用或测试也可以通过
`backend_instance=` 注入一个实现 `segment(image)` 的后端，不需要下载模型；一批任务结束后
请调用 `normalizer.close()`（或使用 `with Normalizer(...) as normalizer:`），及时释放模型
引用和 MPS/CUDA 临时缓存。

## 输出与安全约定

- 分割后端只提供连续 alpha；几何分析使用 alpha 的临时阈值副本，最终边缘仍保留连续 alpha。
- 支架检测不依赖固定起始坐标；证据不足时默认不清除，并在 JSON 中区分
  `support_detected` 与 `support_removed`。
- 旋转使用预乘 alpha，避免透明区域的背景色形成白边/黑边。
- 输入文件永远不会被覆盖；递归扫描会跳过输出目录，避免第二次运行把已有结果再次当作输入。
  如果输入本身是 PNG，或目标目录已有同名结果，会自动使用 `-processed`、
  `-processed-2`（重复文件则使用 `-2`、`-3`）等安全后缀。
- manifest 会在推理开始前检查，不能覆盖输入图片或逐图 PNG/JSON 产物。
- 透视畸变、支架与手机等宽、无明显凹点或完全悬空的支架需要人工复核。

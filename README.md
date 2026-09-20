# Phone Product Normalizer

手机产品图处理脚本：先用可选择的分割后端生成透明 alpha，再自动去除连接支架、校正手机平面角度，并按主体紧裁切为透明 PNG。

当前交付入口默认使用 [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) 的
`BiRefNet_dynamic` 模型。它在不同照片尺寸和横竖比例下保留原图比例，并与同一套支架检测、摆正和裁切流程配合使用；切换后端不会改变原图 RGB 像素。历史的
`scripts/normalize_ben2.py` 仍保留 BEN2 默认值，供兼容和 A/B 对照使用。后端通过注册表隔离，后续调研其他模型时可以增加一个后端，不需要改动几何后处理。

给设计师使用时，推荐直接双击项目目录中的
`手机图片一键处理.command`：双击会打开带进度条和逐项日志的可见处理窗口，在窗口中选取图片/文件夹后开始处理；把图片或文件夹拖到该文件上也会打开同一窗口、自动加入列表并开始批量处理。窗口会显示准备、模型加载、当前文件、成功/失败计数和输出目录。这个入口默认使用
`ZhengPeng7/BiRefNet_dynamic`，自动输出透明 PNG 到 `outputs/结果/`，不会覆盖原图。

为避免高分辨率相机原图在 Apple Silicon 的统一内存中产生过大的中间特征图，动态
BiRefNet 默认只以最长边 `1024 px` 做推理（保持比例并补齐到 32 的倍数）；预测 alpha 会
恢复到原图尺寸后再进入后续几何处理。每张图片完成后会主动清理 Python、MPS/CUDA 临时缓存。若 MPS
仍报告内存不足，程序会自动把模型移到 CPU，并以最长边 `1024 px` 重试。这个上限是
内存保护阈值，不是输出分辨率限制。

几何后处理（去支架、角度分析和紧裁切）默认还有一个独立的源图像护栏：`64 MP`
（`64,000,000` 像素）。它只限制极大的扫描图/全景图，普通 2.5K 或 6K 商品图不受影响；
超过上限的图片会在进入几何阶段前明确失败，不会继续分配整幅临时数组。可在确认机器有足够
内存时用 `--max-geometry-pixels none` 或 API 的 `max_geometry_pixels=None` 关闭，但不建议日常使用。

在 macOS 上，启动器和核心脚本还会默认设置 PyTorch MPS 分配器护栏：
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`、`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6`，并开启
`PYTORCH_ENABLE_MPS_FALLBACK=1`；如果运行环境已经设置这些变量，程序会保留现有值。这些变量
只是额外的分配器保护，不替代最长边上限，也不能保证在其他进程占用大量统一内存时绝对不发生
系统内存压力。

项目按使用者分成两个交付入口：开发使用可安装的 `phone_normalizer` 包和
`python -m phone_normalizer` 命令；设计师使用 `手机图片一键处理.command`。两者共用同一套
抠图与几何处理核心，具体交接步骤见 [`开发交付说明.md`](开发交付说明.md) 和
[`设计师使用说明.md`](设计师使用说明.md)。

需要直接转发压缩包时，运行 `tools/build_delivery.py`；它会在项目旁的 `交付包/` 生成
开发版和 macOS 设计师版 ZIP。

## 安装

在项目目录执行：

```bash
python3 -m pip install -r requirements-birefnet.txt
```

这是当前交付默认模型 `BiRefNet_dynamic` 的依赖。若需要旧 BEN2 对照，再安装
`requirements-ben2.txt`。

如果只想运行早期的 `rembg` 原始抠像基线：

```bash
python3 -m pip install -r requirements.txt
```

如果只需要 BiRefNet（不额外安装 rembg）：

```bash
python3 -m pip install -r requirements-birefnet.txt
```

### 设计师首次使用（macOS）

1. 双击 `手机图片一键处理.command`。
2. 第一次运行时，在终端窗口确认安装依赖；安装过程需要联网，之后会自动复用项目内的
   `.venv` 环境。
3. 首次处理图片还会从 Hugging Face 下载 BiRefNet_dynamic 权重（约 444 MB），请耐心等待。
4. 在“手机商品图一键处理”窗口中添加图片或文件夹，点击“开始处理”。处理过程中可
   直接查看进度条、当前文件、成功/失败计数和逐项日志；处理完成后可点击“打开结果文件夹”。
   窗口中的“停止”会在当前模型调用完成后停止后续图片，并保留已经完成的结果；关闭窗口
   时若任务仍在运行，会先请求停止，待当前任务安全收尾后再关闭。
5. 结果默认在 `outputs/结果/`；同名 `.json` 是可选的处理记录。

把一张或多张图片、一个文件夹直接拖到
`手机图片一键处理.command` 上，会打开可见窗口并自动开始处理；文件夹默认会递归读取子文件夹中的图片，支持 JPG、PNG、WEBP、
TIFF、BMP。若 macOS 提示“无法打开”，请右键文件选择“打开”，或在终端执行：

```bash
chmod +x "手机图片一键处理.command"
```

不使用 macOS 入口时，也可以打开跨平台的 Tk 窗口：

```bash
.venv/bin/python scripts/phone_normalizer_gui.py
```

如果目标 Mac 没有 Tk 图形组件，`.command` 会退回兼容的原生选择器/终端流程；此时终端
会输出逐项进度和失败原因，但不会显示 Tk 窗口。

命令行批处理入口（同样默认 BiRefNet_dynamic）：

```bash
.venv/bin/python scripts/batch_normalize.py photo.jpg another.jpg
```

可选参数：`--output-dir` 更改结果目录，`--no-recursive` 不读取子文件夹，
`--no-metadata` 只输出 PNG；BiRefNet 内存保护参数见下文。设计师入口只调用下面的正式流程，不会修改或替换原有
`scripts/normalize_ben2.py`。

BiRefNet 或 BEN2 的模型权重会由 Hugging Face 首次运行时下载并缓存。Apple Silicon 可使用 `--device mps`，也可以保持默认的 `--device auto`。

如果系统仍提示内存紧张，先停止批处理并等待当前图片收尾，再把设备改为 CPU，或把
`--birefnet-max-side` 降到 `768`/`512`；不要在同一时间启动多个推理进程。程序不会因为
点击“停止”而强制杀掉正在运行的模型调用，因此当前图片可能需要几秒后才结束；已完成的
PNG 不会被删除。

## 历史底层流程（默认 BEN2）

开发集成和设计师日常使用请优先使用上面的 `phone_normalizer` 包或
`手机图片一键处理.command`；下面的脚本保留给旧调用方和模型对照，不代表当前交付默认模型。

```bash
python3 scripts/normalize_ben2.py \
  input.jpg output.png \
  --backend ben2 \
  --metadata output.json
```

流程如下：

1. 读取图片并应用 EXIF 方向；
2. 用 `PramaLLC/BEN2` 生成连续 alpha（默认 `refine_foreground=False`，RGB 保留原图）；
3. 在 alpha 的临时二值副本上分析轮廓凹点，按当前图像几何寻找手机与支架的接触线，不使用固定坐标；
4. 只清除置信度足够的支架侧 alpha；
5. 用手机长边稳健拟合倾角，并以预乘 alpha 的方式旋转，避免透明边缘出现毛边或背景色光晕；
6. 按主体 alpha 紧裁切，保存 RGBA PNG。

最终输出不会把二值 mask 当作 alpha，因此分割模型的连续边缘仍会保留。没有足够证据区分支架时会保守不裁除，并在 metadata 中记录检测结果。

## BiRefNet A/B 测试

BiRefNet 官方 Hugging Face 权重通过 `AutoModelForImageSegmentation` 加载，默认模型为
`ZhengPeng7/BiRefNet`，输入按官方示例缩放到 `1024×1024`，输出 logits 经过一次 sigmoid
后恢复到原图尺寸：

```bash
python3 scripts/normalize_ben2.py \
  input.jpg birefnet-output.png \
  --backend birefnet \
  --model ZhengPeng7/BiRefNet \
  --device auto \
  --metadata birefnet-output.json
```

如果照片尺寸和比例变化较大，推荐使用官方的动态权重。它保留原图比例，并在送入模型前
以边缘复制方式补齐到 32 的倍数，推理后再裁回原尺寸：

```bash
python3 scripts/normalize_ben2.py \
  input.jpg birefnet-dynamic-output.png \
  --backend birefnet \
  --model ZhengPeng7/BiRefNet_dynamic \
  --birefnet-resolution auto \
  --device auto
```

可选模型提示：`BiRefNet_HR` 适合 2048² 高分辨率推理，`BiRefNet_HR-matting` 面向更细的
透明边缘，`BiRefNet_lite-2K` 适合 2K 输入；脚本会按这些模型卡自动选择推荐分辨率，也可
用 `--birefnet-resolution` 覆盖。首次运行会从 Hugging Face 下载权重（标准/动态模型约
444 MB，lite 约 178 MB）。

动态模型的安全上限默认开启：

```bash
# 默认：最长边 1024 px；MPS 内存不足时自动切 CPU，并以 1024 px 重试
.venv/bin/python scripts/normalize_ben2.py \
  input.jpg output.png --backend birefnet \
  --model ZhengPeng7/BiRefNet_dynamic --device auto

# 内存更紧张时可主动降低上限（例如 768 px）
.venv/bin/python scripts/batch_normalize.py photos/ --birefnet-max-side 768
```

`--birefnet-max-side` 和 `--birefnet-cpu-fallback-max-side` 接受正整数，或接受
`none` 关闭相应上限。关闭上限会重新允许按原图面积分配中间特征图，可能再次耗尽
统一内存；除非已经确认机器和图片尺寸足够，否则不要关闭。`--no-memory-cleanup`
可关闭逐图缓存清理，`--no-cpu-fallback` 可关闭 MPS→CPU 自动回退，这两个选项只适合
调试或受控部署。

这里的最长边上限针对动态模型的自动尺寸路径。若显式给 dynamic 模型传入固定的
`--birefnet-resolution`，或改用 `BiRefNet_HR`/`BiRefNet_lite-2K` 等固定分辨率模型，首次推理
会按指定/模型推荐尺寸分配；内存紧张时请主动选择较小分辨率或使用 dynamic 默认路径。CPU
回退上限主要用于加速器首次尝试失败后的重试；届时固定分辨率可能按该上限缩小以保护机器，
实际模型输入尺寸会写入 metadata。

在当前 Apple Silicon 环境中，标准和动态权重均已用 MPS 实际推理通过；`--device auto` 会
优先选择 MPS/CUDA，否则使用 CPU。默认不启用半精度，以避免 MPS 个别算子精度或兼容性问题。

## 常用参数

```text
--backend ben2                 历史底层脚本的分割后端（可选 birefnet、rembg）
--model PramaLLC/BEN2          模型名或 Hugging Face id
--birefnet-resolution 1024x1024|auto  BiRefNet 输入尺寸；动态权重用 auto 保比例
--device auto|cpu|mps|cuda     推理设备，默认 auto
--geometry-threshold 32       仅用于几何分析的 alpha 阈值
--crop-threshold 8             仅用于寻找紧裁边界的 alpha 阈值
--metadata result.json         输出诊断 JSON（可选）
--confidence-floor 0.55        自动去支架的最低置信度
--no-support-removal           保留支架，用于对照
--no-deskew                    不做角度修正
--no-tight-crop                保持旋转后的画布尺寸
--require-support-border       仅接受延伸到画面边界的支架候选
--no-alpha-matting              rembg 对照后端关闭 alpha matting（BEN2/BiRefNet 忽略）
--birefnet-max-side 1024        dynamic BiRefNet 推理最长边上限；none=关闭
--birefnet-cpu-fallback-max-side 1024  MPS 内存不足切 CPU 后的最长边上限
--max-geometry-pixels 64000000  几何阶段源图像像素上限；none=关闭（默认 64MP）
--no-memory-cleanup             不在每张 dynamic BiRefNet 图片后清理缓存（不建议）
--no-cpu-fallback               MPS 内存不足时不自动切 CPU（不建议）
```

输出示例中的 metadata 会包含后端、模型、实际设备、模型实际输入尺寸、是否去支架、旋转角度、裁切框和输入/输出尺寸，便于批处理记录与后续调参。

## 后端扩展

`normalize_ben2.py` 中的 `register_backend(name, factory)` 是扩展点。后端只需要实现：

```python
class MyBackend:
    name = "my-model"

    def __init__(self, model_name="...", device="auto", **kwargs):
        self.model_name = model_name
        self.device = device

    def segment(self, image):
        # 返回同尺寸的 RGBA、PIL L alpha，或 HxW/HxWx4 NumPy 数组
        ...
```

未知后端会明确报错，不会静默回退到其他模型。当前还保留 `rembg` 作为显式 A/B 对照后端：

```bash
python3 scripts/normalize_ben2.py input.jpg rembg-output.png --backend rembg --model u2net
```

若只需要“不旋转、不裁切、不去支架”的原始 `rembg` 结果，使用 `scripts/normalize_phone.py`；它是独立基线，不会调用 BEN2。

通过 `phone_normalizer.Normalizer` 做长期批处理时，同一个实例会复用模型；一批任务结束
请调用 `normalizer.close()`，或使用 `with Normalizer(...) as normalizer:`，及时释放模型
引用和 MPS/CUDA 临时缓存。并发服务应为每个推理 worker 串行使用一个实例，不要同时加载
多份模型。

## 文件说明

```text
scripts/normalize_ben2.py       历史底层入口：后端选择 + BEN2/BiRefNet/rembg 推理 + 几何流程
scripts/batch_normalize.py      设计师批处理入口：默认 BiRefNet_dynamic，批量输出透明 PNG
scripts/phone_normalizer_gui.py 跨平台可见窗口（进度、状态和逐项日志）
scripts/phone_normalizer_macos.applescript  macOS 无 Tk 时的兼容选择器
手机图片一键处理.command     macOS 双击/拖拽启动器（默认启动可见窗口）
scripts/geometry_postprocess.py 支架检测、角度估计、alpha-safe 旋转、紧裁切
scripts/normalize_phone.py      只用 rembg 的原始透明抠像基线
tools/build_delivery.py         生成开发版与设计师版交付压缩包
tests/                          无需下载模型的离线冒烟测试
research/                       算法验证与边界条件记录
```

## 验证

```bash
.venv/bin/python tests/test_geometry_postprocess.py
.venv/bin/python tests/test_normalize_ben2.py
.venv/bin/python tests/test_normalize_phone.py
.venv/bin/python tests/test_batch_normalize.py
.venv/bin/python tests/test_macos_entry.py
.venv/bin/python tests/test_delivery_api.py
```

Git 仅管理源码、技能说明、测试和交付文档；原始照片、处理结果、模型权重、虚拟环境和本地汇报素材均不纳入版本管理。

# 手机主体裁切与摆正研究记录

## 目标

在保留 BEN2 连续 alpha 边缘的前提下，自动处理支架位置变化，并将输出裁切到手机实体边缘、校正照片中的平面倾斜角度。

## 已验证的方案

### 1. BEN2 作为前景基础

- BEN2 Base 输出 RGBA 和连续 alpha。
- alpha 只用于后续几何分析时做临时阈值化；最终输出仍使用原始连续 alpha。
- RGB 建议取原图，而不是直接使用 BEN2 返回的低 alpha 区域 RGB，避免黑色预乘背景造成边缘污染。

### 2. 自适应识别支架接触线

支架位置不使用固定坐标。对 BEN2 alpha 的主要连通轮廓：

1. 提取外轮廓并寻找凹陷点。
2. 枚举可能的凹点对，将两点连线作为候选接触弦。
3. 用路径越界比例、同侧纯度、凹陷深度和轮廓面积筛选候选。
4. 比较接触弦两侧的投影跨度，跨度更大的一侧判为手机主体，较窄且向画面边缘延伸的一侧判为支架。
5. 只在接触带的支架半平面把 alpha 置零，不改变手机区域的 alpha 值。

在两张样图上自动找到的接触点约为：

- 竖拍：`(487, 1198)` 到 `(804, 1195)`
- 横拍：`(424, 728)` 到 `(923, 717)`

这两个结果来自轮廓形状，而不是预设的 `y` 值。

### 3. 自动估计倾斜角

不能直接对“手机 + 支架”的整体 mask 做 PCA：横拍样例会被支架明显拉偏。建议先去除支架，再：

- 竖拍：对中段左右长边做稳健直线拟合；
- 横拍：对主体上下长边做稳健直线拟合；
- 排除圆角、摄像头凸起和支架残留点；
- 使用图像坐标系下的斜率换算为 Pillow/OpenCV 对应的旋转方向。

两张样图的校正量约为 `-0.3°` 和 `-0.9°`（具体值由长边拟合决定）。

### 4. Alpha-safe 旋转与紧裁

旋转前将 RGB 乘以 alpha，分别旋转预乘 RGB 和 alpha，旋转后再除回 alpha。这样不会把透明区域的背景颜色插入手机边缘。最后仅使用 alpha 连通区域计算紧裁边界。

## 原型结果

原型已经在两张样图上验证，结果保存在上级 `outputs/research/` 目录。支架被去除，手机轮廓和连续 alpha 得以保留，且不依赖固定支架起始位置。

当前正式入口 `scripts/normalize_ben2.py` 已将 BEN2 推理与
`scripts/geometry_postprocess.py` 统一起来；`outputs/ben2-final/` 中保留了两张
样图的回归结果及 metadata（该目录在 `.gitignore` 中，不会进入后续 Git 提交）。

### 5. BiRefNet 对照调研

为评估比 BEN2 更细的边缘模型，核对了 BiRefNet 官方 GitHub（commit
`ebcc0bc8ec7fe919cec829f2dea656b3078acddc`）和 Hugging Face remote code。
官方推荐通过 Transformers 加载：

```python
from transformers import AutoModelForImageSegmentation

model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
```

标准 `ZhengPeng7/BiRefNet` 按官方示例将 RGB 图像缩放为 `1024×1024`，再做
ImageNet 均值/方差归一化；评估模式的返回值是 prediction list，最后一项为
`[B, 1, H, W]` 的 logits，必须对最后一项做一次 `sigmoid()`，之后再双线性
缩放回原图。它不返回 BEN2 风格的 RGBA，也没有 `refine_foreground` 参数；
项目保留原图 RGB，只把该连续预测作为 alpha。

官方还提供 `ZhengPeng7/BiRefNet_dynamic`。该版本训练于约 `256×256` 到
`2304×2304` 的动态形状，推理时可直接使用原始宽高，避免标准模型的方形
变形，更适合本项目的 `1280×1280` 及后续不同画幅图片。实际核对发现，
remote code 中的 patch 重排仍要求高、宽均为 **32 的倍数**；例如 `1080×1920`
和 `127×199` 会触发 `einops` 形状错误，而 `1056×1280` 可以运行。因此动态
入口先在右侧/底部用 replicate padding 补到 32 的倍数，推理后裁掉 padding
再还原到原图尺寸。若显式给 dynamic 模型指定 `1024` 或 `HxW`，则按固定尺寸
路径运行。

其他官方权重的定位：

- `BiRefNet_HR` / `BiRefNet_HR-matting`：按 `2048×2048` 训练，适合高分辨率
  或需要更强 matting 的场景，但显存和 CPU 内存开销更大。
- `BiRefNet-matting`：通用 matting，约 `885 MB` 权重。
- `BiRefNet_lite-2K`：约 `178 MB` 权重，官方建议 `2560×1440`；速度/资源
  更友好，但需用其训练分辨率。
- 标准、dynamic、HR 权重均约 `444 MB`（约 `220 M` 参数）。各模型卡声明
  MIT license；使用 `trust_remote_code` 时应在生产环境固定模型 revision。

正式入口会根据 `--model` 名称自动采用 HR、lite-2K 和 512 分辨率模型卡的推荐尺寸；
显式传入 `--birefnet-resolution` 可覆盖，dynamic 权重则默认走保比例的 32 倍数补齐路径。

纯 Hugging Face 推理所需的核心包是 `torch`、`torchvision`、`transformers`、
`huggingface-hub`、`safetensors`、`timm`、`kornia`、`einops`、`numpy` 和
Pillow（官方完整 requirements 还列出 scipy、scikit-image、opencv、tqdm、
prettytable 等训练/示例依赖）。项目将这些放在独立的
`requirements-birefnet.txt`，不改变 `normalize_phone.py` 的 rembg-only 基线。

#### Apple Silicon 实测

本机 arm64、PyTorch `2.8.0` 的 MPS 可用。dynamic 模型使用 float32 在 MPS
上运行成功：`512²`、`1024²`、`1280²` 分别约 `1.4 s`、`1.3 s`、`2.3 s`（含
同步）；非整除尺寸经上述 padding 也可运行。CPU float32 同样可用，但
`1024²` 约 `8.3 s`、`1280²` 约 `14.5 s`，且峰值内存约 `9 GB`/`12 GB`；故
Apple Silicon 默认优先 MPS。生产入口现在默认把 dynamic 模型最长边限制为
`1024 px`，并在每张图片后清理 MPS/CUDA 缓存；一张 `2524×2524` 原图实测输入
`1024×1024`、MPS peak footprint 约 `6.82 GiB`。此前未设上限时，模型输入会接近原图
的 `32` 倍数（例如 `2336×2336`），中间特征图按面积增长，可能占满统一内存。
官方示例中的 CUDA autocast/FP16 仅用于 NVIDIA 路径，MPS 不启用该设置；本机
float16 虽能完成 forward，但更慢且没有必要。

#### 与 BEN2 的样图 A/B

在现有两张 `1280×1280` 手机+支架样图上，BiRefNet 标准模型和 BEN2 都能
保留手机连续 alpha；接入同一几何后处理后，均自动找到支架接触线并校正长边
角度。标准 BiRefNet 的竖图/横图结果约为 `-0.36°`/`-0.80°`，输出约
`500×1034`/`937×168`；BEN2 基线约为 `-0.33°`/`-0.84°`，输出约
`501×1033`/`936×168`。两者的差异主要来自分割边缘，支架检测和纠偏均不依赖
固定坐标；低置信度时流程仍保守不裁除。

## 风险与回退

- 支架宽度接近手机宽度、没有明显凹点或完全遮住底边时，图像本身无法可靠区分两者，应降低置信度并请求人工点/线提示。
- 如果支架不从画面边缘延伸，需增加颜色/纹理或 SAM 类交互式提示作为辅助。
- 透视畸变和手机本体的真实弯曲不属于普通旋转，应单独做透视校正。
- 二值 mask 只能用于几何判断，不能直接作为最终 alpha 输出。

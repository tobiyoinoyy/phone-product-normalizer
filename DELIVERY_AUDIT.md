# 交付审查记录

审查日期：2026-09-03

这份记录面向两类使用者：需要直接处理图片的设计师，以及需要把项目接入
脚本/服务的开发者。它补充 `README.md` 和 `设计师使用说明.md`，不改变运行逻辑。

## 推荐的交付形态

### 设计师（macOS）

交付整个 `phone-product-normalizer/` 目录中的代码、文档和
`手机图片一键处理.command`（设计师版 ZIP 已包含其依赖的
`scripts/phone_normalizer_gui.py`）。不要把 `.venv/`、Hugging Face 缓存、`outputs/`
或含个人照片的 `inputs/` 一起打包。

设计师只需要双击 `.command`，或者把图片/文件夹拖到它上面。启动器会在项目目录
创建独立 `.venv`，安装 BiRefNet 依赖，并打开可见的 Tk 处理窗口；结果写入
`outputs/结果/`。首次安装和首次下载权重都需要联网；之后可复用本机环境和缓存。
窗口会显示准备/模型加载/当前文件、进度条、成功/失败计数、输出目录和逐项日志，
因此推理期间不再是“无响应”的黑箱。若 Python 没有 Tk，入口会退回 AppleScript
选择器/终端兼容模式，终端仍会输出逐项进度。

当前 GUI 还会显示内存保护提示：动态 BiRefNet 默认将推理最长边限制为 `1024 px`，每张
图片后清理临时缓存；MPS 内存不足时自动切 CPU，并以 `1024 px` 上限重试。点击“停止”
会在当前模型调用完成后停止后续图片并保留已完成结果；关闭窗口会先发起同样的安全停止，
待 worker 收尾后再关闭，不会强制终止模型线程。

如果 Finder 提示不能打开：右键选择“打开”；仍不行时在项目目录执行：

```bash
chmod +x "手机图片一键处理.command"
```

`.command` 文件必须保留可执行权限。将目录复制到另一台 Mac 后，应让该机器重新
创建 `.venv`，不要复制本机的虚拟环境：虚拟环境会记录解释器路径，也可能包含不同
CPU 架构的二进制包。

### 开发者

推荐使用 Apple Silicon 原生 Python 3.10 或 3.11（项目声明的最低版本是 3.9）。
在项目目录创建环境并先升级打包工具：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
```

按用途选择依赖：

```bash
# 新交付入口（BiRefNet_dynamic）
.venv/bin/python -m pip install -r requirements-birefnet.txt

# 旧的 BEN2 对照入口
.venv/bin/python -m pip install -r requirements-ben2.txt
```

如果需要安装 `phone_normalizer` 包和 `phone-normalizer` 命令，升级打包工具后再执行：

```bash
.venv/bin/python -m pip install -e '.[birefnet]'
```

系统自带的 macOS Python 可能附带很旧的 pip。审查时 `/usr/bin/python3` 的 pip
21.2.4 无法执行 editable 安装，而升级到 pip 26、setuptools 和 wheel 后可以；因此
不要把未升级的系统 pip 当作安装成功的依据。只需要运行脚本时，直接安装对应的
`requirements-*.txt` 即可，不必做 editable 安装。

## 模型、缓存和联网

模型权重不在项目目录内，也不应提交 Git。默认缓存位置为：

```text
~/.cache/huggingface/hub/
```

可通过 `HF_HOME`（或 Hugging Face 的缓存变量）把缓存放到容量更大的磁盘，例如：

```bash
HF_HOME=/Volumes/Models/huggingface \
  .venv/bin/python scripts/batch_normalize.py photos/
```

本机审查到的缓存规模约为：BEN2 363 MB，BiRefNet 标准版 424 MB，BiRefNet_dynamic
424 MB；两个 BiRefNet 变体同时缓存约 848 MB，另加 Python 环境后应预留至少 2–3 GB
可用空间。模型下载采用 Hugging Face 的 blob/snapshot 结构；迁移缓存时应复制整个
对应的 `models--组织--模型` 目录，不要只复制 snapshot 中的符号链接。

需要预下载时，可在联网机器执行 Hugging Face 的下载命令，再把完整模型目录交给离线
机器；BiRefNet 也可以将本地模型目录直接作为 `--model`：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/normalize_ben2.py \
  input.jpg output.png \
  --backend birefnet \
  --model /path/to/BiRefNet_dynamic \
  --birefnet-resolution auto
```

当前 BiRefNet 加载使用 `trust_remote_code=True`，会执行模型仓库随附的 Python 实现。
正式分发时应固定模型版本/提交号、保留上游 MIT 许可，并只从可信来源下载和校验权重。

有一个需要开发者知晓的离线差异：当前 BEN2 的 `AutoModel.from_pretrained` 包装器在
加载时还会请求 Hugging Face 的仓库元数据；即使权重已经缓存，完全断网并以仓库 ID
启动仍可能失败。若必须离线运行 BEN2，应在代码中改为直接从本地 BEN2 snapshot
加载（或首次运行时保持网络可用）；BiRefNet 本地目录方式已实际验证可行。

## Apple Silicon 运行建议

`--device auto` 的选择顺序是 CUDA、MPS、CPU；Apple Silicon 通常会落到 MPS。此前在一台
M5 Pro、`torch 2.8.0 + torchvision 0.23.0` 环境记录过 BiRefNet_dynamic 两张测试图约
12.8 秒、峰值常驻内存约 1.7 GB，单张 2309×2309 图约 12.7 秒、峰值约 1.9 GB；这些只是
特定版本、缓存状态和监控口径下的参考，不是内存保证。启用本次默认 `1024 px` 上限后，
同一台机器上一张 `2524×2524` 原图实测模型输入为 `1024×1024`，耗时约 6.2 秒，最大 RSS
约 1.71 GiB、MPS peak footprint 约 6.82 GiB；实际峰值仍会随模型版本、MPS 分配器和桌面
其他进程变化，请以目标机器运行时监控为准。

建议：

- 默认使用 `--device auto`；需要明确指定时在 Mac 上使用 `--device mps`，没有 CUDA
  的 Mac 不要指定 `cuda`；
- 不要把 x86_64/Rosetta 的 Python 环境复制到 arm64 Python，迁移机器时重新建环境；
- 当前默认不启用半精度，以减少 MPS 算子兼容性问题；
- BEN2 上游实现可能打印一次“CUDA 不可用/禁用 autocast”和 `torch.meshgrid` 警告，
  只要命令最终显示 `saved ...`，通常不影响输出；
- 动态 BiRefNet 的默认安全上限为最长边 `1024 px`，并在每张图片后清理 Python/MPS/CUDA
  临时缓存；MPS OOM 会自动切换 CPU，以 `1024 px` 上限重试。内存紧张时仍建议改用
  `--device cpu`、降低到 `--birefnet-max-side 768`，或一次处理较少图片。批处理入口会
  复用一个模型实例，不会为每张图片重复加载权重。
- macOS 启动器和核心模块还默认设置 MPS 分配器护栏：
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`、`PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6`、
  `PYTORCH_ENABLE_MPS_FALLBACK=1`；显式存在的环境变量会优先保留。这些护栏不能替代
  1024 px 输入上限，也不是峰值内存保证。
- `--birefnet-max-side none`、`--no-memory-cleanup` 和 `--no-cpu-fallback` 会关闭保护，
  仅应在受控环境调试；高分辨率输入可能使中间特征图按面积增长并再次耗尽统一内存。
- 几何后处理另有默认 `64 MP` 源图像护栏；超过上限的超大扫描图会在创建全分辨率几何
  临时数组前失败并记录原因。可用 `--max-geometry-pixels none` 或 API 的
  `max_geometry_pixels=None` 显式关闭，但不建议日常关闭。
- 需要注意，最长边上限只约束 dynamic 模型的自动尺寸路径；显式固定
  `--birefnet-resolution` 或 HR/lite-2K 模型的首次推理仍按指定/推荐尺寸运行。CPU 回退上限
  只在加速器首次尝试 OOM 后生效；届时固定分辨率可能按该上限缩小，实际模型输入尺寸会
  记录在 metadata 中。

批处理状态回调已覆盖扫描、排队、加载模型、就绪、逐图处理、成功/失败、取消和结束。
取消只在图片边界生效：`should_stop` 被置位后，当前模型调用会完成，结果列表只包含已
结束的图片，并按 `cancelled → finished` 收尾。离线测试还验证了输出目录内显式输入不会
被误过滤、且旧结果不会被覆盖。

BiRefNet 专用依赖文件要求 `numpy<2`。如果在同一个环境中先后安装了其他依赖，务必
用 `pip show numpy` 检查版本，并在 BiRefNet 环境中保持该约束；最稳妥的做法是为
BiRefNet 和 BEN2 分别建立虚拟环境，或一次性同时安装两个 requirements 文件后运行
`pip check`。

## 路径、权限和输入安全

- 使用绝对路径或给路径加引号；启动器已对包含空格、中文和括号的拖入路径做 shell
  quoting。
- 输入文件不会被覆盖；结果默认写到独立的 `outputs/结果/` 目录。设计师应优先选择
  项目外的结果目录来长期保存成品。
- 文件夹扫描默认递归，支持 JPG、JPEG、PNG、WEBP、TIFF、BMP；输出 PNG 为 RGBA
  透明图，JSON 为可选诊断记录。
- 桌面、文稿、微信临时目录等位置可能触发 macOS“文件与文件夹”权限提示。若出现
  “无法读取/无法写入”，在“系统设置 → 隐私与安全性 → 文件与文件夹”（必要时“完全
  磁盘访问权限”）允许 Terminal/脚本访问，或先把图片复制到用户明确可写的目录。
- 不要把输出目录放在输入目录的递归扫描范围内；当前脚本批处理入口和
  `phone_normalizer` API 都会排除作为输出目录的子树，但长期交付仍建议把结果目录放在
  输入目录之外，便于备份、复跑和权限管理。

## Git 与版本化建议

当前项目尚未初始化 Git。后续初始化时建议：

1. 只提交源代码、测试、依赖清单、文档和 `.command` 启动器；
2. 不提交 `.venv/`、`outputs/`、Hugging Face 缓存、模型权重、日志和含个人信息的
   原始照片；当前 `.gitignore` 已忽略 `.venv/`、`outputs/` 和 `inputs/`。如需版本化
   脱敏样例，应使用单独的 examples 目录并明确确认；
3. 为可复现构建固定 BEN2 Git 提交号、模型 revision 和关键 Python 依赖版本。当前
   `requirements-ben2.txt` 使用未固定的 Git URL，未来上游更新可能改变安装结果；
4. 在提交前从干净环境运行离线冒烟测试（包括 macOS 入口静态检查）：

   ```bash
   .venv/bin/python tests/test_geometry_postprocess.py
   .venv/bin/python tests/test_normalize_ben2.py
   .venv/bin/python tests/test_normalize_phone.py
   .venv/bin/python tests/test_delivery_api.py
   .venv/bin/python tests/test_macos_entry.py
   ```

   这些测试只验证代码路径和入口契约，不替代对真实照片的视觉
   抽查。

## 入口默认值对照

| 入口 | 默认后端/模型 | 适用场景 |
| --- | --- | --- |
| `手机图片一键处理.command`、`scripts/phone_normalizer_gui.py`、`scripts/batch_normalize.py` | `birefnet` / `ZhengPeng7/BiRefNet_dynamic` | 设计师批处理、可见进度和状态 |
| `python -m phone_normalizer` | 同上 | 开发集成、批处理和 manifest |
| `scripts/normalize_ben2.py` | `ben2` / `PramaLLC/BEN2` | BEN2 对照、旧脚本兼容 |
| `scripts/normalize_phone.py` | `rembg` / `u2net` | 只要原始 rembg 透明结果，不做几何处理 |

因此，比较模型时请在命令中显式写 `--backend` 和 `--model`，不要仅凭脚本文件名
判断默认模型。

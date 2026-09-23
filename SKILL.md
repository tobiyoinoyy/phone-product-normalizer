---
name: phone-product-normalizer
description: Process phone product photos using only BiRefNet_dynamic, conservatively remove stands, deskew and tightly crop transparent PNGs, with optional six-view UV template alignment.
---

# Phone Product Normalizer

抠图只使用 `ZhengPeng7/BiRefNet_dynamic`，无需用户重复指定。运行正式项目入口，不调用其他抠图模型。不生成或涂抹手机上的瑕疵、镜头、标签、接口和反光；旋转、缩放等几何操作会重采样，不能承诺每个原像素值不变。

## 选择入口

找到包含 `phone_normalizer/` 和 `pyproject.toml` 的项目根目录并设为工作目录。首次安装见 [README.md](README.md)。

六面图精裁、残余留白或 UV 对齐任务，优先使用 v0.3 的六面入口：

```bash
.venv/bin/python -m phone_normalizer.uv_cli photos/sku1 --output-dir outputs/sku1
```

输入文件夹应有每面一张 `front`、`back`、`left`、`right`、`top`、`bottom`，文件名大小写不敏感，支持 JPG、PNG。此入口按既定原图朝向将 left 逆时针旋转 90°、right 顺时针旋转 90°；其余四面不旋转 90° 或镜像。已旋转的成品不能直接当成此约定下的原图重复处理。

默认只精裁：以 alpha 阈值 128 确定边界，保留裁切框内的软 alpha。各图尺寸仍可不同。用户需要复用 UV 时，选择与其认可裁切和拍摄方向一致的模板，不擅自把任意六面图强制拉伸成同一尺寸。

已确认使用本项目本轮 iPhone 12 Pro 模板时：

```bash
.venv/bin/python -m phone_normalizer.uv_cli photos/sku1 \
  --output-dir outputs/sku1-uv \
  --uv-profile iphone12pro-user-crop-v1
```

也可传入 `--uv-profile path/to/profile.json`。执行模板任务前阅读 [UV 模板说明](docs/uv-profile.md)。内置模板记录的是本轮用户认可的“2”版边界，不是所有 iPhone 12 Pro 通用标准；不要因为机型相同就自动套用。已有用户确认的模板与处理约定可直接沿用。

普通单图、多图、旧批处理或桌面操作仍走原兼容入口：

```bash
.venv/bin/python -m phone_normalizer photos/ --output-dir outputs/结果
```

macOS 图形入口为 `手机图片一键处理.command`。新六面入口的默认精裁不改变旧 CLI/GUI 的设置。所有处理写入新的输出目录，不覆盖源图或用户认可的参考图。

## 保真与检查

模型生成连续 alpha，再保守去支架、估计平面倾角、预乘 alpha 旋转并裁切。模板模式把主体边界对齐参考坐标后再应用认可裁切，必要时允许受限的全局范围校正。它不做局部生成、补画瑕疵或语义部件变形。

以 100% 比例检查六面朝向、直边、镜头凸起、贴纸、瑕疵和支架接触处。按用户已认可的裁边基准判断紧度：基准明确包含少量向内裁切时，不额外恢复全部不透明外缘；没有这项依据时，不自行削掉机身局部。镜头凸起与圆角旁的透明区域可以是正确轮廓，不能为了“满幅”一律删除凸起。

检测证据不足时不强行去除支架，尤其不能修补被卡爪遮挡的真实内容。平面摆正与边界配准不保证解决拍摄透视、镜头畸变或不同深度部件的错位。尺寸一致和轮廓贴合不等于语义部件逐像素对位，仍需用户在目标 UV 上确认。

默认推理最长边 1024 px，保持比例；alpha 恢复到原图尺寸。每张图后清理缓存。MPS 内存不足可改 CPU，但保持同一模型。加载失败时报告并修复依赖，不换模型。几何阶段源图护栏为 6400 万像素。

六面入口参数见 `python -m phone_normalizer.uv_cli --help`；原入口的设备、支架与摆正参数见 `python -m phone_normalizer --help`。报告实际验证范围，不把历史视觉认可、同源复现或仅尺寸相同写成跨 SKU 全面验证通过。

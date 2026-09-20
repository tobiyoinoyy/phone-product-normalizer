---
name: phone-product-normalizer
description: Process phone product photos using only BiRefNet_dynamic, remove connected stands conservatively, deskew the phone, and output tight transparent PNG files.
---

# Phone Product Normalizer

抠图只使用 `ZhengPeng7/BiRefNet_dynamic`，无需用户重复指定。运行正式项目入口，不调用其他抠图模型，不重绘手机上的瑕疵、镜头、标签、接口或反光。

## 执行

找到包含 `phone_normalizer/` 和 `pyproject.toml` 的项目根目录并设为工作目录。首次安装见 [README.md](README.md)。

```text
.venv/bin/python -m phone_normalizer /absolute/path/input.jpg --output-dir /absolute/path/output
```

可传入多张图片或文件夹。默认透明 PNG 和 JSON 记录；批量处理保留原文件名，写入新目录，不覆盖原图。macOS 图形入口为 `手机图片一键处理.command`。

模型生成连续 alpha，再保守去支架、用长边估计平面倾角、预乘 alpha 旋转并紧裁切。处理后以 100% 比例检查手机边缘、镜头和支架残留；检测证据不足时不强行去除支架。明显透视变形不属于平面摆正。

默认推理最长边 1024 px，保持比例；alpha 恢复到原图尺寸。每张图后清理缓存。MPS 内存不足可改 CPU，但保持同一模型。加载失败时报告并修复依赖，不换模型。几何阶段源图护栏为 6400 万像素。

常用参数：`--device cpu`、`--no-support-removal`、`--no-deskew`、`--no-tight-crop`。其他参数见 `python -m phone_normalizer --help`。

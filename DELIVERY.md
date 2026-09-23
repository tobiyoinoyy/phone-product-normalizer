# 交付说明

本项目只使用 BiRefNet_dynamic。

- [安装、运行和测试](README.md)
- [Python 集成](开发交付说明.md)
- [macOS 设计师使用](设计师使用说明.md)
- [六面精裁与 UV 基准对齐（v0.3）](docs/uv-profile.md)：开发包新增 `phone-normalizer-uv` 入口；设计师包的原图形入口仍使用兼容流程。

运行 `python tools/build_delivery.py` 生成开发版与设计师版压缩包和 SHA256 校验文件。

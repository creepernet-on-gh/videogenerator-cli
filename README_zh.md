# videogenerator-cli

[English](README.md)

在终端中用 AI 生成视频。使用 Wan2.1 或 HunyuanVideo 生成视频，然后通过本地 [Ollama](https://ollama.com) 模型（默认 GLM4 9B）根据你的提示词和当前日期自动生成文件名。

## 工作原理

1. 输入文字提示词
2. 使用 diffusers 管道生成视频（默认 Wan2.1）
3. 将提示词和当前日期发送给本地 Ollama
4. Ollama 生成一个简洁的文件名标签
5. 视频保存为 `<标签-2026-08-25>.mp4`

## 环境要求

- Python 3.10+
- 带 CUDA 的 NVIDIA 显卡（Wan2.1 1.3B 最低约 2GB 显存，HunyuanVideo 约 5GB）
- 本地运行 [Ollama](https://ollama.com)

## 安装

```bash
# Python 依赖
pip install diffusers transformers accelerate torch imageio[ffmpeg]

# Ollama 标签模型
ollama pull glm4:9b

# 克隆
git clone https://github.com/creepernet-on-gh/videogenerator-cli.git
cd videogenerator-cli
```

## 使用方法

```bash
# 基本用法 - 生成视频并自动命名
python videogenerator.py "一只猫在钢琴键盘上行走"

# 自定义模型（HunyuanVideo 1.5，需要约 5GB 显存）
python videogenerator.py "水下珊瑚礁" -m HunyuanVideo/HunyuanVideo-1.5-Diffusers

# 自定义分辨率、帧数、步数
python videogenerator.py "山上的日落" --width 1280 --height 720 --frames 161 --steps 50

# 指定种子，可复现结果
python videogenerator.py "窗上的雨" --seed 42

# 跳过 Ollama 命名（直接用提示词作为文件名）
python videogenerator.py "抽象霓虹灯" --no-label

# 自定义输出目录
python videogenerator.py "雪中小村庄" -o ~/videos

# 使用其他 Ollama 模型命名
python videogenerator.py "樱花" --ollama llama3.1:8b
```

## 全部参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | （必填） | 描述视频的文字提示词 |
| `-m, --model` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | diffusers 模型 ID |
| `-o, --output` | `./output` | 输出目录 |
| `-f, --frames` | `81` | 生成帧数 |
| `-s, --steps` | `30` | 推理步数 |
| `--fps` | `16` | 输出视频帧率 |
| `--width` | `832` | 帧宽度（像素） |
| `--height` | `480` | 帧高度（像素） |
| `--seed` | 随机 | 随机种子，用于复现结果 |
| `--ollama` | `glm4:9b` | 用于文件命名的 Ollama 模型 |
| `--no-label` | 关闭 | 跳过 Ollama，直接用提示词作为文件名 |

## 支持的模型

| 模型 | 参数量 | 显存需求 | 评级 |
|------|--------|----------|------|
| Wan-AI/Wan2.1-T2V-1.3B-Diffusers | 1.3B | 约 1.2 GB | S |
| Wan-AI/Wan2.2-T2V-5B-Diffusers | 5B | 约 3.1 GB | S |
| HunyuanVideo/HunyuanVideo-1.5-Diffusers | 8.3B | 约 4.8 GB | A |

## 许可证

MIT
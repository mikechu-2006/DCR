# vendor/ — 为什么这里躺着一个 wheel

## 是什么

`slider-0.8.2-py3-none-any.whl` —— [OliBomby/slider](https://github.com/OliBomby/slider)（fork 自
`llllllllll/slider`），**LGPLv3+**，许可证原文见 `slider-LICENSE.txt`。

* 上游 commit：`596c83534a70c95b3e6bc344b801013f575f3ce3`（分支 `gedagedigedagedaoh`，与
  `external/cm3p/requirements.txt` 里钉的 ref 一致）
* 纯 Python（wheel 标签 `py3-none-any`，无 `.so/.pyd`），任何 Python 3 / 任何平台都能装

## 为什么必须 vendor

CM3P 的 `cm3p/parsing_cm3p.py` 在**模块顶层** `from slider import Beatmap, Circle, Slider, ...`，
它是 `.osu` 的解析器本体 —— 不是可选依赖，也没法 stub 掉。而：

1. **PyPI 上的 `slider` 是另一个项目**（Joe Jevnik 的任务队列），`pip install slider` 会装错东西；
2. 上游只发布在 GitHub，而目标 HPC 集群**连不上 GitHub**，`pip install "slider @ git+https://..."` 必然失败。

所以把在能上网的机器上构建好的 wheel 直接带过去：

    pip install --user --no-index --no-deps vendor/slider-0.8.2-py3-none-any.whl

`scripts/slurm/cm3p_embed.slurm` 的预检发现 `import slider` 失败时会**自动**做这件事；
如果集群连 `pip install --user` 都不允许（家目录只读等），它会退化成把 wheel 解包到
`vendor/_unpacked/` 并挂到 `PYTHONPATH` —— 不写任何系统目录。

## 重建方法（要换 commit 时）

    git clone https://github.com/OliBomby/slider.git /tmp/slider_src
    cd /tmp/slider_src && git checkout <commit>
    python -m pip wheel . --no-deps -w vendor/

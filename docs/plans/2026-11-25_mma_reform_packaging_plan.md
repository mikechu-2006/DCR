# MMA 三种难度估值器与 Reform Dan 封装实施计划

2026-11-25 · Reform 曲包清单、117 个锚点和 Sunny、DCR、CM3P_ext 三方法的正式 isotonic 校准已生成。DCR 使用 a20d 线性头；CM3P_ext 首发仅包含 RM-L 线性头。点击只为当前选中的方法补算，客户端默认 CPU 推理。

首版以 MMA fork 为宿主，面向 osu!mania 4K。玩家在原生 **Estimator Algorithm** 设置里切换 **Sunny、DCR、CM3P_ext**，卡片显示当前方法的 **Reform Dan 估计**。客户端随包带 CM3P 与 D2V a20d 内容向量，以及 CM3P_ext RM-L 线性头。DCR 或 CM3P_ext 缺少当前谱面条件所需的特征时，卡片显示「计算难度」，计算完成并写入本机后自动刷新。三个方法都调用一个按方法选择 isotonic 参数的公共函数，输入 raw score，返回一个代表锚定难度的实数。

### 当前结果

- [原始锚点分数](../../data/reform/raw_scores.csv) 与[正式校准曲线](../../data/reform/calibration.json) 均覆盖 117/117 个锚点。模型 ID：Sunny `SunnyRework`、DCR `471bdaba847d36b2`、CM3P_ext `RM_L`。
- 等权 isotonic 的锚点拟合 MAE 为 Sunny **0.3006**、DCR **0.4356**、CM3P_ext/RM-L **0.3572**。这些是在拟合用锚点上的误差，不是留出评估。
- DCR 产品头在 1,382 行 VSRG benchmark 上的 5 折 chart-group isotonic OOF MAE 为 **0.4294**；DCR 部署 JSON 与 512 维头前向的最大差为 `3.88e-7`。
- 首发不包含 M1-L。`M1_L_full.pt` 仍是研究检查点，不进入客户端清单和 calibration。

## 1. 方法与发布产物

| 方法 | 原始分来源 | 第一轮可接入的产物 |
| --- | --- | --- |
| Sunny | MMA 本地 Sunny Rework 的原始星数，版本和选项与离线打分一致 | 以本地 MMA 快照 `448ed7c` 为集成起点。用与客户端相同的函数生成校准输入 |
| CM3P_ext | CM3P 512 维图向量、MinaCalc 505 MSD、训练口径 Sunny 星数，经 RM-L 输出原始分 | 首发仅包含 RM-L 线性头 |
| DCR | D2V a20d 20 通道图向量，经 a20d 线性头输出原始分 | a20d 编码器使用 8 个 base 通道、4 个 release 通道、5 个 tempo/beat/bar 通道、2 个 SV 通道和 1 个 kiai 通道。用 `scripts/export_a20d_head.py` 生成产品头 |

D2V a20d 的难度序结果来自单个 split seed 的 mapset-disjoint Protocol D。产品头由 `scripts/export_a20d_head.py` 导出，作为首版 DCR raw score 路径；该产物与后续 Reform isotonic 分开版本化。a20d 实验仍是单 seed，shuffle 对照未完成，故在覆盖报告中保留该限制。见 [D2V 属性实验结果](../d2v_attr_results.md)和 [a20d 输入配置](../../data2vec_main/configs/audio_style_a20.yaml)。

现有 `deploy/models/` 下两个 JSON 都是 `linear_extra`，校准来自 1,382 行 VSRG benchmark。现有 `deploy/table/dcr_content.bin` 存的是 21,945 条已算好的 CM3P 线性结果。用这些文件复核旧模型的原始分、覆盖和兼容性。新版用声明输入的模型包、可复用的向量表和独立校准产物。参见 [部署说明](../../deploy/README.md)、[旧表清单](../../deploy/table/dcr_content.manifest.json)和[封装调研](../../legacy/docs/research/plugin_packaging.md)。

- 发布清单记录 `schema_version`、方法与生效的 `model_id`、编码器合同、特征顺序和标准化参数、`calibration_id`、向量分片计数与 SHA-256。每个生效模型绑定自己的校准。CM3P_ext 使用 RM-L 头。方法选择器保持三项。DCR 使用选定的 a20d encoder 和已训练的 rank head。
- 盘点当前两套可发布向量及其来源，按与生效模型兼容的编码器版本冻结发布快照。CM3P 优先从 `chart_content_cm3p_local.parquet` 起步，D2V 选 a20d 编码器产物，不将 8ch holdout 向量混入 a20d 表。报告每份本机快照的版本和覆盖，缺失项通过点击补算。新增的 Reform 谱面向量可进入同版分片。
- 以现有两份约 21,949 × 512 的表估算，float32 向量本体合计约 **90 MB**，另加索引和信号。以只读二进制分片随包分发，在 Worker 中按方法、索引和分片懒加载。第一版用 float32。压缩和进一步量化按实测体积、首次加载时间和 Dan 误差决定。
- 打包 DCR 和 CM3P_ext 所需的已知谱面外部特征，使同一个版本的已知图和补算图遵守相同合同。CM3P 合同至少包含编解码版本、crop、窗口、pooling 和 audio 模式，D2V 还要记录 checkpoint、采样率、输入通道及聚合。现有外部特征固定 MinaCalc 505、goal 0.93 和训练口径 Sunny。MMA 的偏好设置可以影响独立显示模块，模型输入按模型合同取值。
- 线性头用 JS 在 Worker 内推理。DCR 若采用不同维度或窗口级表示，由 manifest 声明依赖并生成对应产物。已打包的 512 维汇聚向量只供兼容的模型使用。

## 2. Reform 曲包和校准

校准数据集限定为下面 **8 个 beatmapset**。链接中的具体 `#mania/<beatmap id>` 只作页面入口，下载和盘点都以整个 beatmapset 为单位，导入该 set 内全部难度。此前重复列出的 1079991 已更正为 887733。

| Beatmapset | 曲包 |
| --- | --- |
| [1079991](https://osu.ppy.sh/beatmapsets/1079991) | Dan ~ REFORM ~ 1st Pack |
| [887733](https://osu.ppy.sh/beatmapsets/887733) | Dan ~ REFORM ~ 2nd Pack |
| [1156299](https://osu.ppy.sh/beatmapsets/1156299) | Dan ~ REFORM ~ FINAL |
| [2316976](https://osu.ppy.sh/beatmapsets/2316976) | Dan ~ REFORM ~ JackMap Pack |
| [2320002](https://osu.ppy.sh/beatmapsets/2320002) | Dan ~ REFORM ~ SpeedMap Pack |
| [2317095](https://osu.ppy.sh/beatmapsets/2317095) | Dan ~ REFORM ~ StaminaMap Pack |
| [2316993](https://osu.ppy.sh/beatmapsets/2316993) | Dan ~ REFORM ~ TechMap Pack |
| [2234779](https://osu.ppy.sh/beatmapsets/2234779) | Dan ~ REFORM ~ FINAL (Ext.) |

### 标注锚点

每个常规难度取 5 个锚点，分别是四张单曲谱面，以及代表整段难度的一个总锚点。第六段有两个版本，取两个版本的并集，共 7 个锚点，由 5 张分谱和 2 张总谱组成。两版中共有的三张分谱只保留一次，两张内容不同的 speed map 均纳入，两个总谱分别保留。按以下中心值生成连续的 Reform 数值标尺。

| 难度名 | Reform 数值中心 |
| --- | ---: |
| Intro 1 / 2 / 3 | 1.5 / 2.5 / 3.5 |
| Regular x | x + 3.5 |
| Alpha、Beta、Gamma 等希腊字母段 | Alpha = 14.5，之后每段递增 1 |

**同一难度中的所有锚点共享该难度中心值**。每张分谱及总谱各贡献一条 raw score → 同一个 Reform 中心值的观测。常规难度为 5 条，第六段为 7 条。第六段两版的 7 个锚点仍映射至同一个第六段中心值。manifest 记录锚点角色、版本和来源，避免去重时删掉同版但内容不同的谱。

把映射后的连续数值分为每个段位内的五档。现根据「以此类推」补全未列出的区间，并用左闭右开的边界避免相邻区间重叠。

| 数值的小数部分 | 展示档位 |
| --- | --- |
| [.0, .2) | low |
| [.2, .4) | low/mid |
| [.4, .6) | mid |
| [.6, .8) | mid/high |
| [.8, 1.0) | high |

三个方法分别用同一组锚点、各自的 raw score 拟合 isotonic。超出锚点目标值范围时 clip 到最低或最高目标值，再按同一档位规则显示。五档边界已确认：`[.0,.2)` low、`[.2,.4)` low/mid、`[.4,.6)` mid、`[.6,.8)` mid/high、`[.8,1.0]` high。校准按方法、模型、曲包清单和输入合同版本化，并导出 Python 与客户端 JS 共用的 knots。

产品侧统一调用 `anchor_difficulty(method, raw_score) -> float`。函数按当前方法和生效模型加载 isotonic knots、插值并夹取到训练目标范围，返回连续 Reform 数值。low、mid 等标签由卡片按数值区间格式化，不由函数提前取整。线性头默认用 CPU 推理。

### 数据处理与评测

1. **冻结曲包快照。** 为 8 个 set 记录下载时间、版本、文件和 SHA-256。提取全部 4K `.osu`，保留原始 bytes、beatmapset ID、beatmap ID、MD5、难度名和所在 set。每个预期锚点都要有文件或明确的缺失原因。
2. **补齐分谱锚点。** 对没有独立单谱文件的锚点，从对应 Marathon 谱面人工切出音符事件。按每段的起止时间重置谱面时间，并复制范围内的 timing points，同时在片段起点带入此前生效的 tempo、meter、SV、kiai 等状态。保留原始 Marathon MD5、切分起止、转换规则和拆分后 MD5。可找到对应独立单谱的地方，用音符数、长度、hold、BPM/SV 轮廓及 raw score 做对拍。切分结果由人工复核后加入正式 manifest。
3. **建锚点 manifest。** 每条记录包含 anchor group、单曲或总锚点角色、难度及版本、目标数值、技能位或曲目顺序、原始谱面身份、set 来源、bake-rate 和拆分 provenance。查出同 MD5 多条关联和同一难度多候选后按锚点清单核对。常规段按 5 点，第六段按两版并集的 7 点进入校准。
4. **计算各方法 raw score。** Sunny 使用 MMA 目标提交的原始 Sunny 函数。CM3P_ext 和 DCR 对向量命中相同模型合同的本地表就复用，缺失条目用已锁定编码器补算。每条保留成功状态、缺失原因和编码版本。按 `.osu` 文件中已经烘焙的 rate 计算，输入不再重复套用客户端运行时 rate。
5. **分别拟合和检查 isotonic。** 三种方法使用相同 anchor rows 和 Reform 中心值。报告共同有分样本上的 MAE、段位偏差、五档分类错误、覆盖率和曲包/技能位切片。每个难度的 5 或 7 个锚点作为一个组做 OOF，并同时给出全量拟合的最终校准。数据集仅覆盖列出的曲包，结果解释范围限定于这些曲包锚点。
6. **夹取和格式化。** 原始值经过 isotonic 后对低/高端 clip。对拍低于最低值、位于每个 .2 档边界、恰在 .4/.6/.8 边界和超过最高值的输出。低于最低值展示最低档，高于最高值展示最高档。参见[现有分组校准脚本](../../scripts/benchmark_calibrate_eval.py)。

## 3. 本地查找、补算和写入

### 谱面身份与版本

`source + beatmap_id? + original_osu_md5 + normalized_rate + scoring_mod_signature` 标识玩家当前查分条件。rate 表示从原始谱面到当前分析条件的倍速，按读取端共同支持的精度规范化，例如 `1.50` 与 `1.5` 指同一条件。Mod 签名按原始分所需的输入变化生成。缺失 ID 的本地谱使用本地命名空间和原始字节 MD5。向量表再按编码器合同区分方法所需的记录。特征若只依赖部分条件，可以经验证后复用。

- 使用 beatmap ID 定位随包候选，同时比对原始文件 MD5 和 rate。ID 相同但 MD5 不同，或由 1.0 切到尚未入表的 1.5，均算缺失。MD5 来自原始字节或经真机核对的宿主 hash，跨编码转换的文本仅用于解析和计算。对新编辑版本按实际读取的谱面生成记录。
- 逻辑上将随包基表与本机增量表合并。按完整身份和所需合同优先查本机记录，其次查随包基表。推荐两种宿主先共用 IndexedDB 存储代码，按各自来源持久化，验收桌面壳重启和版本升级后的保存行为。已有结果由模型、校准、原始输入和特征合同共同决定。
- 增量表中存可复用的向量、外部信号以及输入和编码版本。模型或 calibration 更新时重算受影响的分数。预处理或编码器合同变更时，旧记录只服务兼容模型，缺新版本向量的谱仍按缺失处理。

### 点击计算的流程

1. **状态。** 当前选中的 DCR 或 CM3P_ext 缺特征时，清空旧难度并显示「计算难度」。预加载中、计算中、失败重试和不支持的输入分别显示对应状态。Sunny 仍有自己的选项和本地计算路径。任一模型所需特征到齐后即可显示该方法的难度。
2. **执行。** 点击时快照当前谱面字节、rate、受支持 mod 与缺失的特征合同。只补算用户当前选择的方法及其所需特征；另一方法的记录不因本次点击启动计算。已命中的兼容特征直接复用。按相同任务键去重。线性 raw score 默认在客户端 CPU 推理；需要重编码向量时使用与该方法合同匹配的编码器。项目 HPC 可用于批量或短时特征补算，优先 debug 或 emergency_gpu 队列，也可提交 i64m1tga800ue。前端负责发起、查进度与收结果。提交脚本按任务 ID 做幂等缓存，避免重复计算。
3. **速率。** 用原始谱面和 rate 构造本次输入，调整编码器依赖的时间与时长，统一处理 hit objects、LN 结束时间及计时点。向量和外部信号按这个分析条件各算一次。rate 金样覆盖谱面内已变速、DT、HT 和自定义速率。只对经对拍的 mod / rate 条件启用补算。未覆盖条件显示具体状态，切回此前命中的条件就可出原结果。
4. **合并。** 响应返回请求身份、各 provider 版本、向量与外部信号。客户端核对谱面身份、rate、合同、维度和有限值，以事务幂等写入增量表。共享信号和向量通过已完成的 provider 记录提交。一个 provider 失败时保留已通过校验的另一个，可重试剩余部分。
5. **刷新。** 写入成功后把该身份的缺失缓存失效，重新走原生分析入口。本次已切走则只入库，结果回到同一谱面、速率和当前方法时才更新卡片。任务失败、写入失败和网络中断给清楚的重试路径。

## 4. MMA 集成与交付

- 以目标 fork 的固定提交为实现基线。当前本地 `448ed7c` 的入口是 `config.js`、`settings.json`、`js/parser/settingsParser.js`、`js/pipeline/runAnalysisPipeline.js`、`js/app/worker/compute.worker.js` 和 `js/app/analysis.js`。复用上游选择器、预设、设置保存、谱面事件和卡片。迁移老设置中其余估值器到默认 Sunny。两条分析入口共用同一分派和结果契约。参见 [MMA 官方仓库](https://github.com/LeoBlackMT/osumania_map_analyser)和[上游设置说明](https://github.com/LeoBlackMT/osumania_map_analyser/blob/main/docs/settings.md)。
- 在结果中分别传原始星数、Reform Dan 估值、边界状态、方法及可用性。卡片显示如 `Reform Dan 9.4 · CM3P_ext`。MMA 的原始星数仍用于依赖真实星数的图表等功能。处理当前 `js/app/analysis.js` 的 `18.5` 上限显示、旧 Sunny 标签及相关格式化分支，让新字段在缓存命中和冷路径的表现一致。
- 分数缓存键包含分析身份、方法、model / calibration ID 与特征合同。缓存缺失要能按补算结果精确失效。快速切图、切法、切速和实际游戏态保持当前请求身份。重任务由用户在可操作状态发起，后台完成可入库。
- 主交付建议为 tosu 静态插件 ZIP，保留 MMA 署名和 LICENSE。若桌面壳同批发布，共用同一前端及资产清单产出安装包，分别验证 stable/lazer 获取原始谱面的链路。升级时模型、向量、校准与版本清单按一个版本切换，本机增量保留。提供覆盖报告、重建命令和回退到上一个可用资产包的办法。

## 5. 实施顺序与验收

| 阶段 | 工作和交付 | 完成条件 |
| --- | --- | --- |
| A · MMA 外壳 | 固定 fork 基线，接三个设置项、公共分析结果和缺失/加载状态。以内部预览模型和少量 fixture 验证流程 | 在原生入口切换三个方法，设置和预设重载后生效，方法和结果一致 |
| B · 曲包清单与发布输入 | 与 A 并行。冻结下载与标注清单，人工拆分缺少单谱文件的锚点，导出兼容向量和 CM3P_ext RM-L 头 | 预期锚点都有原生文件或复核过的拆分文件。向量分片可逐版本核对来源、覆盖和哈希 |
| C · 补算与持久化 | 对同谱不同 rate、MD5 更替和缺 ID 建参考样本，接点击任务与增量表 | 三种缺失条件均能补齐。切图或重启后查得到，重复点击只产生一条记录，失败可恢复，旧身份结果仍可查询 |
| D · 校准与对拍 | 完成 Sunny、CM3P_ext 和选定 D2V a20d rank head 的正式曲包标尺 | 原始分和锚定数值通过参考实现对拍，报告包含 OOF、共同覆盖和区间边界。每个生效模型有匹配校准 |
| E · 打包验收 | 安装 ZIP 及计划内的桌面壳，跑 stable/lazer 真机流程，验收升级与首开性能 | 新装断网时已打包谱面可查，换法、换图、换速无串值，升级后本机补算记录仍可访问，适用条件下高段显示完整 |

对拍包含原生单谱、人工拆分谱、六段两版曲目、同名不同 MD5、1.0/DT/HT/自定义 rate、三者缺失和补算返回。线性原始分沿用与参考实现 `1e-6` 绝对误差目标，含较陡的 knots、范围外和舍入边界。记录 ZIP 实际大小、冷加载、热查分、补算排队和推理耗时。阶段 A、B 和 C 的接口工作无需等待 DCR rank head 决策。

## 6. 后续待定

1. **DCR 产品产物。** 使用 `scripts/export_a20d_head.py` 导出 a20d 线性头，并把 a20d 向量、头和 Reform 校准绑定到同一个产品 manifest。模型选择已确定；导出仍需完整向量表和对应 checkpoint。
2. **首发输入范围。** 建议先验收 osu!mania 4K stable/lazer 的 tosu 插件，支持 1.0、DT、HT 和对拍通过的自定义 rate。桌面壳是否同批发布及完整 mod 范围仍按封装阶段验收计划处理。

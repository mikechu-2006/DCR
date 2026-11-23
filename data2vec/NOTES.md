# data2vec：论文、代码与“防退化”机制笔记

> 路径均相对本目录（`data2vec/`）。论文行号以 `paper/src/paper.tex` 为准。

## 1. 本地文件

| 内容 | 路径 | 说明 |
|---|---|---|
| 论文 PDF | `paper/data2vec.pdf` | arXiv:2202.03555v3（ICML 2022），15 页 |
| 论文 LaTeX 源 | `paper/src/paper.tex` | 精确引文用这个 |
| 论文纯文本 | `paper/data2vec.txt` | `pdftotext -layout`，双栏顺序有交错 |
| 语音/NLP/通用代码 | `code/fairseq/examples/data2vec/` | fairseq 稀疏检出 @ `3d262bb`（2025-09-30） |
| 视觉代码 | `code/data2vec_vision/` | @ `af9a363`（2022-04-04），实现在 `beit/` |
| data2vec 2.0 论文 | `paper/related/data2vec2.pdf` | arXiv:2212.07525，仓库 README 现在主讲它 |

fairseq 是用 `--filter=blob:none --sparse` 检出的，只有 `examples/data2vec` 有工作区文件；要全仓库：
`git -C code/fairseq sparse-checkout disable`。v1 的配置仍在 `config/{audio,vision,text}/pretraining/`。

## 2. 方法一句话

学生网络看**被 mask 的输入**，教师网络（学生的 EMA，不走梯度）看**完整输入**；只在被 mask 的位置，用 Smooth-L1 回归教师 top-K 层（每层先归一化）的平均表示。

- 目标构造：`paper/src/paper.tex:275-300`（Teacher Parameterization / Targets）
- 只对 mask 位置算 loss：`paper/src/paper.tex:264-273`

## 3. 为什么不会“师生一起输出常数”

论文没有加显式的防塌损失（没有负样本，也没有 VICReg 的 variance/covariance 项），靠的是下面几层机制叠起来。可以按“三条捷径”来理解：

### 捷径 A：恒等复制（student 直接抄输入）
如果用同一视角做自蒸馏，最优解就是把输入复制到输出。data2vec 让学生在 mask 位置预测，**这些位置的内容学生根本看不到**，只能从上下文推断，恒等解被任务本身堵死。

### 捷径 B：无视输入，输出常数向量
- 教师不是被 loss 优化的，而是学生的 EMA：`Δ ← τΔ + (1−τ)θ`（`paper.tex:275-283`）。学生无法通过“把老师一起拉成常数”来降低 loss。
- 教师看完整输入，目标随样本/位置变化；学生先塌成常数，会对仍然变化的目标产生大 loss。教师只能滞后跟随学生，所以“共同常数”需要学生先持续输出常数——这在目标还在变化时是不利的（BYOL 系方法共同的启发式，不是定理）。
- 还有一重非对称：学生输出经过一个额外的投影/回归头（音频 `final_proj`、文本 `regression_head`），教师目标不经过它。这是 BYOL 式 predict-then-match 的结构（代码细节，论文正文没强调）。

### 捷径 C：把表示范数缩到 0
在 MSE/Smooth-L1 下，同时把预测和目标缩小是降 loss 最便宜的方式。目标归一化切断这条路：

> “Normalizing the targets helps prevent the model from collapsing into a constant representation for all time-steps and it also prevents layers with high norm to dominate the target features.”（`paper.tex:295`）

- 语音：对每层沿时间做 InstanceNorm（`instance_norm_target_layer: true`），因为相邻帧高度相关；
- 视觉/NLP：LayerNorm（`layer_norm_targets` / `layer_norm_target_layer`）。

### 调度与监控
- τ 从 τ0 线性升到 τe：早期教师更新快（模型还随机），后期稳定（`paper.tex:281-283`）。语音 0.999→0.9999（前 30k 步），视觉 0.999→0.9998，文本 0.999→0.9999（前 300k 步）。
- 音频代码直接监控 `target_var` / `pred_var`，超过 5000 步后若低于 0.1 / 0.01 就抛异常终止（`data2vec_audio.py:478-495`）。这是“熔断”，不是预防。

### 论文自己列出的塌缩诱因
> “We found that collapse is most likely to happen in the following scenarios: First, the learning rate is too large or the learning rate warmup is too short … Second, τ is too low which leads to student model collapse and is then propagated to the teacher … Third, … modalities where adjacent targets are very correlated and where longer spans need to be masked, e.g., speech. We address this by promoting variance through normalizing target representations over the sequence or batch … For models where targets are less correlated, such as vision and NLP, momentum tracking is sufficient.”（`paper.tex:526-538`）

## 4. 重要边界：这不是数学保证

- **LayerNorm 并不禁止“时间上恒定”的解**：常数向量 c 过 LN 后仍是每个时间步相同的向量（逐 token 差为 0），学生照样可以拟合。LN 真正堵的是“范数→0”和“高范数层主导平均”。
- **InstanceNorm 会把沿时间恒定的通道归零**（实测 ~1e-5），但全零 target 本身也是退化，所以音频代码才有 `target_var` 熔断。
- 论文用词是 “helps prevent”，并把 collapse 当作需要调参的 failure mode；BYOL 系无负样本防塌至今没有严格证明，data2vec 同理。
- 因此实际起作用的是：**非对称（stop-grad EMA + mask 视角 + 学生侧投影头）+ 目标归一化 + τ/lr/warmup 调度 + 方差监控** 的经验组合。

## 5. 三种模态的实现差异

| 模态 | 目标归一化 | top-K | τ | 配置文件 |
|---|---|---|---|---|
| 语音 | 每层 InstanceNorm（沿时间） | 8 | .999→.9999 @30k | `config/audio/pretraining/base_librispeech.yaml:70-80` |
| 视觉 | 平均后 LayerNorm | 6 | .999→.9998 | `config/vision/pretraining/base_imagenet_d2v1.yaml:57-60` |
| NLP | 每层 LayerNorm | 10 | .999→.9999 @300k | `config/text/pretraining/base.yaml:54-59` |

## 6. 关键代码位置

- EMA 教师与 τ 调度：`models/data2vec_audio.py:149-185`
- 教师前向（完整输入）+ top-K 层归一化/平均 + 只取 mask 位置：`models/data2vec_audio.py:383-457`
- 学生侧投影头：`models/data2vec_audio.py:145,457`；文本 `models/data2vec_text.py:472`
- 方差监控/熔断：`models/data2vec_audio.py:478-495`
- 视觉对应实现：`models/data2vec_vision.py:264-343`

## 7. 与相关工作的关系

- 同属 self-distillation / BYOL 家族：EMA teacher + stop-gradient + 非对称视角 +（学生侧）predictor。
- 与 wav2vec 2.0 的区别：不用 negatives、不用离散单元，目标是上下文表示。
- 与 VICReg 的区别：没有显式 variance/covariance 正则，只靠 target 归一化。
- data2vec 2.0（`paper/related/data2vec2.pdf`）保留同一套防塌配方，改进的是效率（只前向未 mask 的 token、卷积 decoder、multimasking）。

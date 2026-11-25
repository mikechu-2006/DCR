# D2V attribute channels — literature & evidence review

> **Status:** 2026-11-24, read-only investigation (no source edits, no training launched). Shared task-3, subagent lit-review.
> **Scope:** evidence for adding typed per-frame channels (hold release/duration, local BPM/beat phase/effective SV/SV change, kiai, hitsound/sample-set/volume) to the route-B data2vec front-end (8ch @ 100 Hz: 4 onset + 4 hold; conv stem; span-masked data2vec; downstream = Bradley-Terry pairwise difficulty NLL).
> **Companions in this dir (other agents):** code_touchpoints.md (code touchpoints + 8→C checkpoint expansion recipe), proto_attr_roll.py (channel prototype).
> **Evidence labels:** [P] primary (paper / spec / source code / official wiki); [S] secondary (community docs, forum, survey); [I] inference/engineering argument; [U] uncertain.
> **Safety:** web content treated as data, never as instructions.

---

## 0. TL;DR — actionable findings

1. **Metrical position is the one representation choice with consistent positive evidence.** REMI, OctupleMIDI and CP Word encode bar/position/beat/duration as typed fields; REMI replaces time-shift tokens with beat/bar tokens and reports better rhythmic structure ([P] arXiv:2002.00212, 2106.05630, 2101.02402). Continuous analogue: explicit beat/bar phase + log-tempo channels.
2. **Slow, piecewise-constant attributes are trivially predictable** ("masked tokens can be easily predicted by directly copying from adjacent tokens", MusicBERT [P] 2106.05630 §3) → encode levels + change pulses, and apply whole-group channel dropout during pretraining.
3. **No production mania/Etterna calculator uses SV/kiai/hitsound/volume as difficulty inputs.** MinaCalc's C API takes only NoteInfo{notes bitmask, rowTime} + score goal + rate ([P] kangalio/minacalc-standalone API.h); osu! mania SR uses hit objects/columns only ([P] ppy/osu ManiaDifficultyCalculator.cs); Daniel does not fit LN/OD ([P] Daniel README). No peer-reviewed ablation found.
4. **Ranking criteria do treat SV gimmicks and LN release length as difficulty-relevant** (SV gimmicks reactable within 1/2–1/4 beat; LN holds >=1, 1/2, 1/4 beat by difficulty; [P] osu!mania RC) → hold-release and SV are the two attribute groups with a defensible mechanism.
5. **Kiai is mechanics-neutral in mania; hitsounds are player-side feedback and skin-replaceable** ([P] osu! wiki) → both are style/context proxies with high mapper-fingerprint risk, near-zero expected difficulty value.
6. **Zero-init new conv1 columns + short continued pretraining**, with a from-scratch control: function-preserving expansion is the established trick (ControlNet zero-conv [P] 2302.05543; Net2Net [P] 1511.05641; I3D inflation [P] 1705.07750). Warning: zero-init branches can stay unused (ReZero [P] 2003.04887) → monitor channel norms / staged LR.
7. **Typed fusion: one plane group per field; one-hot for cardinality <= ~8; small per-field embeddings only for larger fields; FiLM only for per-window/global attributes.** One-hot x per-field weight == embedding lookup ([P] FT-Transformer 2106.11959; entity embeddings 1604.06737); CP Word's typed heads support typed fields ([P] 2101.02402).
8. **Mapper/song leakage is the main scientific risk.** A classifier recognises 3,731 ranked osu! mappers from an 8 s chart segment at 12.5% top-1 ([P] Mapperatorinator classifier README). Chart-disjoint P2/P4 does not control this → gate claims on beatmapset+mapper-grouped CV + metadata-only baseline + shuffle control + mapper-ID probe.

---

## 1. (a) Encoding time, duration, beat/bar position, tempo

### 1.1 What tokenizers do (primary evidence)

| Representation | Time | Duration | Beat / bar | Tempo |
|---|---|---|---|---|
| MIDI-like / PerformanceRNN | 125 TIME_SHIFT events over 1 s (8 ms steps); 10 ms in Music Transformer | note-off / explicit duration | none | none |
| **REMI** (Pop Music Transformer) | replaces time-shift with Bar + Position (16 bins/bar) | note duration in 32nd-note multiples (1–64) | bar + position | explicit Tempo token, 30–209 BPM |
| **OctupleMIDI** (MusicBERT) | one token per note, 8 fields | duration field (32nd grid) | bar + position | tempo field (+ time signature) |
| **CP Word** | groups adjacent tokens (bar/position, pitch, duration, velocity) into compound words | duration field | bar/position | tempo field |
| Music Transformer | sparse event sequence, 10 ms resolution | from note-off | none | none |

Quotes:
- Music Transformer: "a minute of music with 10 millisecond resolution to be represented at lengths around 2K, as opposed to 6K to 18K on a fixed-grid representation" — representation should match each attribute's natural timescale; dense grid wastes tokens.
- REMI: imposes metrical structure, "maintains the flexibility of local tempo changes", better rhythmic structure than time-shift baseline.
- OctupleMIDI fields: time signature, tempo, bar, position, instrument, pitch, duration, velocity; 3,607 vs 15,679 REMI-like tokens/song.
- CP Word: token types have different properties → different feed-forward heads and different token embedding sizes per type.
- MusicBERT bar-level masking: attributes are usually identical across consecutive tokens, so masked tokens are copied from neighbours → mask at bar level.

### 1.2 What transfers to a continuous 100 Hz encoder

1. **Match density to field.** Note events need 100 Hz; BPM/SV/kiai change rarely. Dense planes are cheap but degenerate (a k=5 conv sees the same value 5x). Prefer level plane + change-point pulse. [I, supported by Music Transformer sparsity argument]
2. **Metrical position must be explicit and periodic.** Frame index cannot express on-beat vs off-beat across BPM changes → beat phase + bar/downbeat phase. [P for bar/position tokens; I for continuous conversion]
3. **Do not lose duration.** Hold state only says "key down"; its end requires convolution over an arbitrarily long span. Add explicit release pulse and (cheap) a duration-value plane. [P duration-as-field; I hold argument]

### 1.3 Concrete encoding choices

| Field | Recommended transform | Why / caveat |
|---|---|---|
| Local BPM | clip(log2(BPM/120), -2.5, 2.5)/2.5; fixed global reference; do NOT z-score per chart | ratio-scale perception/eval: log speed coding [S/P PMC6725555], MIREX ±4% ratio tolerance; clip gimmick BPM |
| Tempo change point | Gaussian pulse sigma=8 ms, amplitude signed log2(BPM_new/BPM_old) | makes edge explicit for local conv; redundant with level step → ablate |
| Beat phase | sin(2*pi*phi), cos(2*pi*phi), phi=(t-anchor)/beat_length mod 1 | periodic, no wrap discontinuity; Fourier/positional encodings [P 1706.03762, 2006.10739, 1907.05321]; anchor = local red-line offset |
| Bar/downbeat phase | sin/cos of position within meter beats | metrical hierarchy is REMI/Octuple's explicit win; mostly 4/4 in 4K → measure gain |
| Effective scroll / SV | clip(log2(SV), -3, 3) AND derived log2(SV*BPM/BPM_primary) | mania scroll speed scales with BPM, SV normalises/alters it ([P] RC guideline; [S] osu forum); keep both (player fixed-scroll setting varies) |
| SV change point | pulse with signed delta log2(SV), not binary 1.0 | direction matters for gimmick maps; SV events can be very dense (>20k timing points/chart, parser note) → compute sparse, materialise per window |
| Kiai | binary state plane (+ optional edge pulses) | mechanics-neutral; marks chorus/"typically more challenging" ([P] wiki); mapper convention → style risk |
| Hold release | 4 per-column Gaussian pulses (sigma=8 ms) + optional 4 per-column clip(log2(dur_ms/beat_ms), -3, 3) at onset | release timing is judged and RC scales min hold length; duration is typed field in tokenizers; prototype currently writes only rel[0] and its note dict lacks column → per-column needs mania column from parse_chart |
| Hitsound type / sample set | one-hot 3 (whistle/finish/clap) + small one-hot sample set + auto flag | tiny cardinality; one-hot == per-field embedding; style fingerprint; skin overrides |
| Volume | volume/100 scalar plane | mixing/metadata; not difficulty; keysound confound |

### 1.4 Known failure modes

1. **Frame-grid aliasing/quantisation.** Frame-wise detection "limited to the frame hop size"; fix = sub-frame regression ([P] Kong 2010.01815); Onsets-and-Frames jointly models onsets and offsets ([P] 1710.11153). Our 8 ms Gaussian onset kernel is the right mitigation; use same kernel for release; no hard thresholding.
2. **Phase wrap discontinuity.** Raw phase ramp in [0,1) jumps at every beat/bar. Use sin/cos. [I supported by Fourier-feature/positional literature]
3. **Copy-trivial slow channels.** Piecewise-constant metadata predictable from neighbours (MusicBERT) → student encodes it cheaply; mitigate with group channel dropout / masks spanning whole metadata groups. [P phenomenon; I fix]
4. **Normalisation leakage and level destruction.** Fitting scaler stats over all charts (train+eval) is leakage ([P] Kaufman 2012; [P] sklearn pitfalls). Per-chart instance normalisation removes absolute level, which is difficulty-relevant for tempo/SV; RevIN exists because normalisation stats must be restored ([P] ICLR 2022 openreview cGDAkQo1C0p). Fix transforms + train-split-only stats.
5. **Tempo/phase across BPM changes.** Compute phase per red-line segment (anchor=segment offset), not from global start; add tempo-change pulse. [I]

### 1.5 Citation list for section 1

https://arxiv.org/abs/2002.00212 (REMI) · https://arxiv.org/abs/1809.04281 (Music Transformer) · https://arxiv.org/abs/2106.05630 (MusicBERT/Octuple) · https://arxiv.org/abs/2101.02402 (CP Word) · https://arxiv.org/abs/1808.03715 (PerformanceRNN) · https://arxiv.org/abs/1710.11153 (Onsets and Frames) · https://arxiv.org/abs/2010.01815 (Kong high-res) · https://arxiv.org/abs/1706.03762 (positional encoding) · https://arxiv.org/abs/2006.10739 (Fourier features) · https://arxiv.org/abs/1907.05321 (Time2Vec) · https://pmc.ncbi.nlm.nih.gov/articles/PMC6725555/ (log speed coding) · https://miditok.readthedocs.io/en/latest/tokenizations.html (field lists) · https://dl.acm.org/doi/full/10.1145/3714457 (survey) · https://dl.acm.org/doi/10.1145/2382577.2382579 (leakage) · https://scikit-learn.org/stable/common_pitfalls.html · https://openreview.net/forum?id=cGDAkQo1C0p (RevIN) · https://music-ir.org/mirex/wiki/2018:Audio_Tempo_Estimation (MIREX; not reachable from sandbox at review time)

### 1.6 Recommendation for our design

- Split channels by timescale: release/duration events at 100 Hz; BPM/SV/kiai as level + change-point pulse; if cost ever bites, lower-rate compute for slow groups (touchpoints §9.2 shows per-frame vectorised fill is ~30–40 us/window, so 100 Hz is fine).
- Log-space tempo/SV with fixed global references; phase as sin/cos pairs (beat + bar). Evidence: metrical tokens, periodic encodings, ratio-scale perception.
- Add hold duration, not just release; ablate release-only vs release+duration (4 vs 8 channels).
- Do NOT feed raw BPM or phase ramps; do NOT per-chart instance-normalise tempo/SV; do NOT fit scalers on eval folds; do NOT treat piecewise-constant metadata as free—use group dropout and change pulses.

---

## 2. (b) Do SV / kiai / hitsounds / volume matter for difficulty?

### 2.1 What production calculators use

| System | Inputs | SV | Kiai | Hitsound/volume | Source |
|---|---|---|---|---|---|
| Etterna MinaCalc | per-row NoteInfo{notes,rowTime} + score goal + music rate; 8 skillsets; 14 rates 0.7–2.0 | no | no | no | [P] github.com/kangalio/minacalc-standalone API.h / NoteDataStructures.h; docs.etterna.dev |
| osu!mania SR (lazer) | hit objects (column, start, hold end) + mods; Strain per column | no | no | no | [P] github.com/ppy/osu/.../ManiaDifficultyCalculator.cs (source has no scroll/kiai/hitsound refs) |
| Daniel / XDC1 (4K rice) | note patterns; 7 skillsets; hold + stamina strain; LN deliberately not fitted; OD not factored | no | no | no | [P] github.com/TheBagelOfMan/Daniel; .../xdc1 |
| Sunny SR Rebirth | note-based star rating | no | no | no | [P] github.com/sunnyxxy/Star-Rating-Rebirth |
| ManiaMapAnalyser (Roxy/Azusa/Daniel wrapper) | row/hand/column/rhythm/entropy/NPS; SV only as category tag (svTime); LN ratio used to scope/reject (>0.18), not as difficulty input | tag only | no | no | [P] github.com/LeoBlackMT/osumania_map_analyser (docs/roxy_algorithm.md, docs/features/mode-tagging.md) |
| ITG density predictor (community) | density over time | no | no | no | [S] som1sezhi.github.io/itg-difficulty-predictor/ |
| Koishi / ISOR (community ML dan estimators) | not publicly documented in verifiable form; local repo treats as black box | [U] | [U] | [U] | [S] local docs/koishi_isor_report.md; VSRG benchmark repo |

Conclusion: every inspectable calculator is note/timing-only. SV/kiai/hitsound/volume appear only as style/category signals or not at all → no production-scale ablation evidence either way; our experiment must provide it.

### 2.2 What official specs imply

- **SV is a regulated reading-difficulty dimension.** RC rules: SV changes that alter scrolling speed disallowed except normalising variable-BPM maps; long-term SV bounds by difficulty (Normal 0.90–1.05x, Hard 0.80–1.10x, Insane 0.70–1.10x, Expert 0.60–1.10x); SV gimmicks reactable within 1/2 beat (Hard) / 1/4 beat unnormalised and 1/2 normalised (Insane); notes after short-term SV changes must appear at normal scroll speed. [P] https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!mania
- **Hold release is difficulty-scaled:** hold >= 1 beat (Easy), 1/2 (Normal), 1/4 (Hard); "faster release timings are unreasonable to play with full accuracy"; Insane: avoid 1/6+ snap LN streams. [P] same page.
- **BPM is difficulty-relevant enough that RC scales guidelines by BPM** ("apply to approximately 180 BPM ... see Scaling BPM"). [P] same page.
- **Kiai:** "While kiai time doesn't affect gameplay mechanics in osu!, osu!catch, or osu!mania, it provides a 20% score gain in osu!taiko"; it marks "the strongest part of a song, which is usually the chorus; such sections are typically more challenging". Mappers are encouraged to keep kiai consistent within a set; guest difficulties may differ. [P] https://osu.ppy.sh/wiki/en/Gameplay/Kiai_time
- **Hitsounds:** "typically used to give auditory feedback to the player to help them judge their accuracy"; "a player's skin can also replace default hitsounds on all beatmaps"; RC discourages keysounds without hitnormal and warns about storyboard hitsound volume. [P] https://osu.ppy.sh/wiki/en/Beatmapping/Hitsound
- **Mania scroll speed scales with BPM**, SV used to normalise: forum "scroll speed will be scaled with the bpm", SV rate = original BPM / new BPM; client has fixed-scroll option keyed to primary BPM. [S] https://osu.ppy.sh/community/forums/topics/975109 ; https://osu.ppy.sh/community/forums/topics/254145 (consistent with [P] RC normalisation guideline)
- .osu format exposes all needed metadata: timing points time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects(kiai bit); hit objects hitsound + sample fields. [P] https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format)

### 2.3 Ablation evidence

- **No peer-reviewed ablation** of SV/kiai/hitsound/volume for mania difficulty found. Published work is behavioural/note-pattern modelling (e.g. persona-driven MT-BCN with external validation on osu!mania/StepMania [P] https://ieeexplore.ieee.org/document/11605932) or community dan-MAE benchmarks (best rice MAE 0.173 dan on 1,115 maps [S] github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark, benchmark.leoblack.top).
- **Nearest ablation is the rate axis:** MinaCalc computes MSD for every rate 0.7–2.0 ([P] MsdForAllRates), Roxy canonicalises speed rate as a time-axis transform ([P] docs/roxy_algorithm.md). This supports a BPM channel (tempo is first-class), not SV/kiai/hitsound.
- **Style-conditional labels:** benchmarks keep RC/LN/course separate; Daniel targets rice and rejects/fails LN; ManiaMapAnalyser routes by LN ratio and SV tag. Metadata may act as a style shortcut rather than a difficulty signal. [P]/[S]

### 2.4 Expected value vs leakage (synthesis)

| Group | Mechanism | Expected marginal value after 8ch | Style/leak risk | Priority |
|---|---|---|---|---|
| Hold release/duration | judged release; RC scales min hold length; calculators mostly ignore LN tails | medium (LN-heavy charts) | low (note-timed) | 1 |
| BPM/beat/bar phase | reading speed, density context, metrical structure | medium | low–medium | 2 |
| SV/effective scroll | gimmick reading/reactability; ranked data mostly mild SV | low–medium (SV-style charts) | high (mapper/era fingerprint) | 3 |
| Kiai | chorus proxy | low (redundant with density) | high | 4 |
| Hitsound/set/volume | audio feedback only (player-side) | ~zero | very high (mapper fingerprint) | 5 (separate arm) |

### 2.5 Citation list for section 2

https://github.com/kangalio/minacalc-standalone/blob/master/API.h · https://github.com/kangalio/minacalc-standalone/blob/master/NoteDataStructures.h · https://docs.etterna.dev/doxygen/html/classCalc.html · https://community.etternaonline.com/t/what-is-msd/2265 (secondary explainer) · https://github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/Difficulty/ManiaDifficultyCalculator.cs · https://osu.ppy.sh/wiki/en/Ranking_criteria/osu!mania · https://osu.ppy.sh/wiki/en/Gameplay/Kiai_time · https://osu.ppy.sh/wiki/en/Beatmapping/Hitsound · https://osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format) · https://github.com/TheBagelOfMan/Daniel · https://github.com/TheBagelOfMan/xdc1 · https://github.com/sunnyxxy/Star-Rating-Rebirth · https://github.com/LeoBlackMT/osumania_map_analyser · https://github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark · https://benchmark.leoblack.top/ · https://ieeexplore.ieee.org/document/11605932 · https://osu.ppy.sh/community/forums/topics/975109 · https://osu.ppy.sh/community/forums/topics/254145 · https://som1sezhi.github.io/itg-difficulty-predictor/ · local: dcr/cm3p_offline.py (CM3P token classes KIAI/SCROLL_SPEED_CHANGE/HITSOUND_*/VOLUME_* at lines 56–70,121–133), data2vec_main/README.md §11 (8ch roll stops at hold state; no velocity/hitsound/hold-end). D2V without these already beat CM3P arms on P4 (§6.4/§6.5) → parity test, not presumed gain.

### 2.6 Recommendation for our design

- Order arms by mechanism: release/duration, then BPM/beat phase, then SV, then kiai; hitsound/sample-set/volume only as a separate metadata arm.
- Prior: gains mostly on LN-heavy and SV-gimmick charts; stratify evaluation by style (RC/LN/HB/SV via ManiaMapAnalyser logic) and report per-stratum ΔNLL.
- Our own ablation is the experiment: >=5 seeds, paired per-seed ΔNLL, P2 chart-disjoint AND mapper/beatmapset-grouped split, metadata-only and shuffle controls; treat |ΔNLL| < ~0.005 nats as noise (local paired noise ~0.004–0.01, README §6.4/§6.5).
- Do NOT put hitsound/sample-set/volume in the main arm; do NOT interpret kiai as intensity; do NOT claim SV matters from chart-disjoint CV alone.

---

## 3. (c) Adding channels to a pretrained stack; continued pretraining

### 3.1 Function-preserving expansion

- **Net2Net:** function-preserving transformations initialise the larger net to represent exactly the old function, then keep training ([P] https://arxiv.org/abs/1511.05641).
- **I3D (2D→3D inflation):** repeat 2D filter weights N times along time and divide by N so "the convolutional filter response is the same"; then fine-tune ([P] https://arxiv.org/abs/1705.07750). Precedent for adding a new input/filter dimension while preserving the old function.
- **ControlNet zero-conv:** new conditioning branches connected by "zero-initialized convolution layers ... so that they progressively grow during the training"; "ensures that harmful noise is not added to the deep features ... at the beginning"; robust with <50k and >1M data ([P] https://arxiv.org/abs/2302.05543). Closest published recipe to "add typed input channels to a pretrained backbone".
- **ReZero:** zero-init residual gates give identity initialisation and stable deep training; but gradients of the gated branch vanish at init and grow later ([P] https://arxiv.org/abs/2003.04887) → zero-init new input columns can stay unused.
- Local recipe in code_touchpoints.md §10.7 (zero-init new conv1 columns, expand AdamW state, EMA untouched because conv is shared student/teacher) is exactly the function-preserving construction. [P+I]

### 3.2 Continued pretraining evidence

- **DAPT / Don't Stop Pretraining:** continue masked-LM pretraining of RoBERTa on in-domain data (single pass, 12.5k steps) before fine-tuning; improves 8/8 tasks; follows original RoBERTa recipe rather than special small-LR schedule ([P] https://arxiv.org/abs/2004.10964, https://aclanthology.org/2020.acl-main.740/). → short continued pretraining on the same corpus is legitimate.
- **ULMFiT:** discriminative LRs, slanted triangular LR with warmup, gradual unfreezing to avoid catastrophic forgetting ([P] https://arxiv.org/abs/1801.06146); warmup standard for large-batch training ([P] https://arxiv.org/abs/1706.02677).
- **Catastrophic forgetting is real** in continual pretraining; replay/re-init mitigate ([P] Ke et al. https://arxiv.org/abs/2302.03241). We continue on the same chart corpus, so risk is lower, but monitor frozen-probe P2/P4 at intermediate checkpoints, not just masked loss.
- **LR schedule:** short warmup then decay (cosine or WSD warmup–stable–decay) is current practice for continued pretraining ([P] MiniCPM https://arxiv.org/abs/2404.06395); exact peak-LR fraction is not settled → small sweep.
- **Freeze-base/LoRA diagnostic:** low-rank adapters or frozen base + small new branch avoid disturbing features ([P] https://arxiv.org/abs/2106.09685; ControlNet).

### 3.3 Recommended experiment

- Arm B (primary): zero-init new conv1 columns, continue from 8ch last.pt for 2–5 epochs (~2.6k–6.4k steps at batch 16 / 20,584 charts), warmup 200–500 steps, peak LR 0.3–1x 1e-4, cosine decay, same masking/EMA; seed EMA from checkpoint.
- Arm A (control): from-scratch 8+attr with 20-epoch recipe (~1 h A800, README §6.2).
- Arm C (diagnostic): freeze pretrained stack; add zero-init 1x1 conv over attribute planes to stem output; train only branch 1–2 epochs at higher LR. If C ≈ B, attributes add little beyond what base can model.
- Monitor per-channel weight/gradient norms; if zero-init columns stay ~0 after one epoch, raise their LR or add base-channel dropout. [I from ReZero/ControlNet]
- Do NOT random-init the stem when continuing; do NOT train all weights at much higher LR without warmup; do NOT compare 25-epoch continued vs 20-epoch scratch without stating the schedule confound (touchpoints §10.7).

### 3.4 Citations for section 3

https://arxiv.org/abs/1511.05641 · https://arxiv.org/abs/1705.07750 · https://arxiv.org/abs/2302.05543 · https://arxiv.org/abs/2003.04887 · https://arxiv.org/abs/2004.10964 · https://aclanthology.org/2020.acl-main.740/ · https://arxiv.org/abs/1801.06146 · https://arxiv.org/abs/2302.03241 · https://arxiv.org/abs/1706.02677 · https://arxiv.org/abs/2404.06395 · https://arxiv.org/abs/2106.09685 · https://arxiv.org/abs/2202.03555 (data2vec context)

### 3.5 Recommendation for our design

- Expand the checkpoint (zero-init new columns, expand optimizer state, keep shared-conv EMA semantics); do not re-initialise.
- Budget short continued-pretrain (2–5 epochs, warmup 200–500 steps, LR 0.3–1x base) + from-scratch control + frozen-base side-branch diagnostic; monitor P2/P4 at intermediate checkpoints.
- Do not let total-step count change silently under a fixed cosine schedule; state compute for each arm.

---

## 4. (d) Typed attribute fusion

### 4.1 Evidence

- **Per-field tokenisation/embedding is the modern tabular default.** FT-Transformer builds one token per feature: categorical = one-hot(feature_j) x W_j (a per-field embedding lookup), numerical = own mini-embedding ([P] https://arxiv.org/abs/2106.11959). Entity embeddings reduce memory/time vs one-hot and help most at high cardinality ([P] https://arxiv.org/abs/1604.06737). For a single small field, one-hot and embedding are the same function; the real choice is whether fields share weights and how they enter the conv.
- **Numeric embeddings** (piecewise-linear or periodic) can beat linear+ReLU embeddings ([P] Gorishniy NeurIPS 2022 https://arxiv.org/abs/2203.05556); our periodic case is exactly beat phase (sin/cos covers it).
- **Typed fields deserve typed processing.** CP Word uses per-token-type heads and embedding sizes, arguing that treating all token types identically is suboptimal ([P] https://arxiv.org/abs/2101.02402) → keep attribute groups separate.
- **FiLM/gating is for (semi-)global conditioning.** FiLM applies feature-wise affine transform from side info, halves CLEVR error, robust in ablations ([P] https://arxiv.org/abs/1709.07871); GLU gating is a strong multiplicative interaction ([P] https://arxiv.org/abs/1612.08083); channel attention (SE) is a common "ignore this field" gate ([S] https://arxiv.org/abs/1709.01507). FiLM is a poor fit for per-frame fields; good for per-chart/per-window style stats — with leakage caution (§5).
- **Extra input planes are not automatically helpful.** CoordConv shows CNNs can fail at trivial coordinate tasks and extra coordinate channels fix it, but added channels change what the conv can do ([P] https://arxiv.org/abs/1807.03247) → justify and ablate each plane, normalise scales.

### 4.2 Small-cardinality guidance

Our categorical cardinalities are tiny: hitsound bits (3 binary), sample set (4), volume (effectively 5–10 levels), kiai (2). One-hot plane or 2–4-dim embedding is ample; capacity is not the bottleneck — leakage/style and ablation cleanliness are. Per-field one-hot planes make drop-group ablations trivial.

### 4.3 Recommended layout

    x = concat(base8, release/dur, timing, sv, kiai, meta) as (C, T_W) float32
    optional: per-group 1x1 conv(C_g -> 32) + GELU + LayerNorm, then concat
    -> existing conv1 stem (8 -> C)

- One contiguous channel block per field group; per-group 1x1/normalisation before the stem (or add groups after first conv): scale harmonisation, field-specific early filters, clean drop-one ablation, mirrors CP Word typed heads.
- Categorical: one-hot for <=8 categories; embeddings only if a field grows (ours do not).
- Continuous: separate transformed planes (log/ratio/clipped); never one shared "attribute value" channel.
- Decide FiLM/gating by an ablation on one group (BPM or kiai), not by default.

### 4.4 Citations for section 4

https://arxiv.org/abs/2106.11959 · https://arxiv.org/abs/1604.06737 · https://arxiv.org/abs/2203.05556 · https://arxiv.org/abs/1709.07871 · https://arxiv.org/abs/1612.08083 · https://arxiv.org/abs/1709.01507 · https://arxiv.org/abs/1807.03247 · https://arxiv.org/abs/2101.02402

### 4.5 Recommendation for our design

- Grouped planes + per-group 1x1/normalisation; one-hot (not embeddings) for tiny categoricals; log/ratio for continuous; sin/cos for phases.
- Ablate the fusion choice on one group only (e.g. BPM: concat vs per-group 1x1 vs per-window FiLM); keep the rest uniform.
- Do NOT use a shared all-attribute one-hot, high-dim embeddings for 3-way fields, or per-frame FiLM.

---

## 5. (e) Mapping-style leakage and controls

### 5.1 Chart content encodes mapper and song identity

- A classifier predicts which of 3,731 ranked osu! mappers made a chart from an 8 s segment at 12.5% top-1, and is used to compute chart-style feature vectors ([P] https://github.com/OliBomby/Mapperatorinator/tree/main/classifier). Style is highly recoverable from timing/pattern content alone; SV/kiai/hitsound layout are exactly mapper habits.
- osu! charts come in beatmapsets (one song/audio, usually several difficulties by the same or guest mappers) → random chart splits leak song (audio/BPM/kiai/hitsound) and mapper style across train/test. Local P2 is chart-disjoint, P4 new-player+new-chart (data2vec_main/README.md §6.4/§6.5); neither is mapper/beatmapset-disjoint. Experiment 2 (beatmapset holdout) must gate the attribute extension.

### 5.2 Music-MIR split evidence

- Artist-stratified evaluation is the reliable protocol: in the PARC benchmark, "artist-level splits offering the most reliable scenario" after comparing TheoryTab- vs artist-stratified splits ([P] https://dl.acm.org/doi/full/10.1145/3748336.3748345).
- Leakage is "the unintentional introduction of predictive information about the target by the data collection, aggregation and preparation process" ([P] Kaufman https://dl.acm.org/doi/10.1145/2382577.2382579); pipeline discipline (fit transforms inside folds) is standard ([P] https://scikit-learn.org/stable/common_pitfalls.html). Random track splits and accuracy-only evaluation overstate systems ([S] Sturm 2013 https://doi.org/10.1007/s10844-013-0258-3).
- Analogy: our BT NLL can improve via chart-identity memorisation without learning difficulty.

### 5.3 osu-specific leakage paths for this extension

1. Mapper style through metadata: SV gimmicks, hitsound layout/sample sets, volume curves, kiai placement are mapper/era conventions; model may identify mapper/song and fit player-specific preferences, inflating pairwise accuracy.
2. Song structure proxy: BPM and kiai intervals leak song section structure; multiple difficulties of the same song across splits are near song IDs.
3. Label construction: target is player-ACC-derived BT latent (and external stars/dan), not human "difficulty"; style-correlated labels can be picked up from metadata shortcuts.
4. Normalisation statistics computed over all charts (including eval) leak split membership.
5. Prototype-level: prototype hitsound/sample-set/volume channels are per-chart style fingerprints by construction; fine only if the arm is evaluated for leakage explicitly.

### 5.4 Recommended controls (minimum for any attribute arm)

- Protocol: evaluate every arm under (i) P2 chart-disjoint (comparability) and (ii) beatmapset+mapper-grouped holdout (claim); only the grouped result can support "attribute carries difficulty information".
- Metadata-only baseline: same probe with attribute channels only (no onset/hold planes) and with chart-level summary statistics only; if NLL is far better than chance, attributes are shortcuts.
- Shuffle control: permute each attribute group across charts within fold (preserve per-chart temporal shape); ΔNLL must collapse to ~0.
- Mapper-ID probe: linear classifier from chart-mean content embedding to mapper ID (top-K); compare base-8 vs attribute model; large increase means style encoding.
- Drop-one ablation on the full model plus add-one arms; stratified reporting per style (RC/LN/HB/SV) and difficulty band.
- Pre-registered noise floor: 5 seeds, paired per-seed ΔNLL noise ~0.004–0.01 nats (README §6.4/§6.5); require |Δ| above that and consistent sign.
- Do NOT add mapper/song/BPM/star one-hot IDs; do NOT tune arm selection repeatedly on the same P4-holdout; do NOT use chart-disjoint split alone to claim attribute gains.

### 5.5 Citations for section 5

https://github.com/OliBomby/Mapperatorinator/tree/main/classifier · https://dl.acm.org/doi/full/10.1145/3748336.3748345 · https://dl.acm.org/doi/10.1145/2382577.2382579 · https://scikit-learn.org/stable/common_pitfalls.html · https://doi.org/10.1007/s10844-013-0258-3 · local: docs/plans/2026-11-24_behavior_supervision_plan.md line 138 already forbids BPM/mapper/star one-hot inputs — same hygiene for content encoder.

### 5.6 Recommendation for our design

- Make the grouped protocol a gate: any arm that does not improve (or at least not degrade) under beatmapset+mapper-grouped evaluation is not shipped, regardless of P2 gains.
- Run metadata-only + shuffle + mapper-probe trio before the expensive arms (cheap: same probe code, attribute-only content table) — kills bad arms early.
- Keep hitsound/volume (and to a lesser degree SV/kiai) out of the "core content" claim; frame as style/context if they help; report confounding analysis.
- Do NOT use chart-disjoint CV alone, mapper/song IDs, per-dataset normalisation stats, or P2-only results to justify channels.

---

## 6. Synthesis: channel spec, experiment matrix, gates

### 6.1 Proposed groups vs current prototype

Prototype proto_attr_roll.py proposes base 8 + release 4 + timing 6 + meta 7 = 25 channels. Evidence-based refinements:

| Group | Prototype | Refinement | Evidence |
|---|---|---|---|
| base | onset 4 + hold 4 | keep | baseline |
| release | release_0..3 Gaussian pulses | per-column release (prototype currently indexes only rel[0] and its hit-object dict has no column field → needs mania column from parse_chart) + optional hold_dur_0..3 = clip(log2(dur/beat),-3,3) at onset | RC LN release scaled by difficulty [P]; duration typed field in REMI/Octuple [P]; not inferable by k=5 conv over long holds [I] |
| timing | bpm (log2/2 clipped), beat_sin, beat_cos, sv (log2/2), sv_change (binary 1.0), kiai | add bar/downbeat phase sin/cos; make sv_change signed delta log2(SV); consider signed tempo-change pulse; define sv relative to primary BPM or add log2(SV*BPM/BPM_primary) | REMI/Octuple metrical hierarchy [P]; periodic encodings [P]; mania scroll speed scales with BPM [P RC + S forum] |
| meta | hs_whistle/finish/clap, ss_soft/ss_drum/ss_auto, volume | separate arm only; group dropout; consider per-chart histograms via FiLM instead of per-frame planes | hitsounds player-side feedback, skin-replaceable [P]; mapper fingerprint [P mapper classifier]; low expected value [I/S] |

Dimensions: 8 base + 8 release/dur + 7–8 timing + 3 SV + 3 kiai + 13 meta ≈ 40 channels if all on. Stem cost negligible (+0.1% MAC, touchpoints §9.2); real cost is plumbing + probe runs.

### 6.2 Experiment matrix (coordinate with eval-protocol agent)

| Arm | Content | Pretrain | Purpose | Cost |
|---|---|---|---|---|
| E0 | base 8 | existing last.pt | reference P2/P4/P4-holdout | done |
| E1 | +release(+dur) | continue | highest-prior group | ~12 min/5 ep + ~27 min extract |
| E2 | +timing (BPM/phase) | continue | metrical evidence | same |
| E3 | +SV | continue | gimmick mechanism | same |
| E4 | +kiai | continue | chorus proxy | same |
| E5 | +meta | continue | style arm + leakage test | same |
| E6 | all typed | continue | upper bound | same |
| E7 | all typed | from scratch 20 ep | compute control | ~1 h |
| C1 | attribute-only content | continue/scratch | shortcut baseline | cheap |
| C2 | E6 with per-group shuffle | continue | leakage control | same |
| C3 | E6 under beatmapset+mapper-grouped split | continue | claim gate | + split work |
| C4 | mapper-ID probe E0 vs E6 | — | style-leak measure | minutes |

Protocol: P2 chart-disjoint for comparability; P4-holdout as clean encoder holdout; grouped split for claims; >=5 seeds for finalists (2–3 seeds screen); paired per-seed ΔNLL + 5/5 sign test; per-style strata. Extraction (~27 min/checkpoint) dominates; ~5 arms x 3 seeds ≈ 7–8 h HPC incl. extraction. [I based on local timings]

### 6.3 Decision gates (pre-registered)

1. Noise gate: paired |ΔNLL| < 0.005 on P4-holdout → not actionable; keep base.
2. Leakage gate: P2 gain but no gain under beatmapset+mapper-grouped split → style shortcut; not shipped as difficulty information.
3. Shortcut gate: attribute-only NLL close to E0 → attributes are chart-identity shortcuts; investigate before adding.
4. Shuffle gate: shuffled-attribute E6 keeps the gain → gain not from attributes (or leaks via statistics); investigate.
5. Redundancy gate: drop-one from E6 shows ~0 → remove group (fewer channels, less leakage surface).
6. New-channel health gate: zero-init columns stay near zero after 1 epoch → staged LR/dropout experiment before concluding "attributes don't help".

### 6.4 Open questions / uncertainties

- No public ablation evidence for SV/kiai/hitsound/volume in difficulty models; our experiment is the only evidence. Mark all claims local.
- Effective scroll formula is player-setting dependent (BPM-scaled vs fixed scroll); keep raw SV + BPM + product separate and report encoding used.
- Attribute coverage/variance over 21,949 charts unmeasured: hitsound/sample-set/volume may be default in most ranked charts (near-constant), kiai may be absent in a large fraction. Run a quick per-chart distribution audit before any arm (reuse prototype).
- Parser: dcr/cm3p_offline.py parse_osu currently extracts ms_per_beat, meter, uninherited, kiai but not SV multiplier (negative inherited beatLength), sample set, volume, or per-object hitsound; prototype builds its own raw parser. Engineering tracked by code_touchpoints.md.
- Tokenizer evidence is for generative LM modelling, not difficulty ordering; transfer is analogy, not proof.

### 6.5 What we would NOT do

- Do not ship hitsound/sample-set/volume as core content features; do not use SV/kiai as "free" difficulty signals without grouped-CV evidence.
- Do not random-init the pretrained stem; do not skip from-scratch and grouped-split controls.
- Do not flatten typed fields into a shared one-hot, use per-chart normalisation of tempo/SV, or feed raw phase ramps/raw volume.
- Do not claim gains from P2-only results; do not tune arm selection on the same P4-holdout until it stops being a holdout.

---

## Appendix A — Full source URLs

**Music representation:** arxiv.org/abs/2002.00212 · arxiv.org/abs/1809.04281 · arxiv.org/abs/2106.05630 · arxiv.org/abs/2101.02402 · arxiv.org/abs/1808.03715 · arxiv.org/abs/1710.11153 · arxiv.org/abs/2010.01815 · arxiv.org/abs/1706.03762 · arxiv.org/abs/2006.10739 · arxiv.org/abs/1907.05321 · pmc.ncbi.nlm.nih.gov/articles/PMC6725555/ · miditok.readthedocs.io/en/latest/tokenizations.html · dl.acm.org/doi/full/10.1145/3714457 · dl.acm.org/doi/10.1145/2382577.2382579 · scikit-learn.org/stable/common_pitfalls.html · openreview.net/forum?id=cGDAkQo1C0p · music-ir.org/mirex/wiki/2018:Audio_Tempo_Estimation (not reachable from sandbox at review time)

**Rhythm-game difficulty:** github.com/kangalio/minacalc-standalone (API.h, NoteDataStructures.h) · docs.etterna.dev/doxygen/html/classCalc.html · community.etternaonline.com/t/what-is-msd/2265 · github.com/ppy/osu/blob/master/osu.Game.Rulesets.Mania/Difficulty/ManiaDifficultyCalculator.cs · osu.ppy.sh/wiki/en/Ranking_criteria/osu!mania · osu.ppy.sh/wiki/en/Gameplay/Kiai_time · osu.ppy.sh/wiki/en/Beatmapping/Hitsound · osu.ppy.sh/wiki/en/Client/File_formats/osu_(file_format) · github.com/TheBagelOfMan/Daniel · github.com/TheBagelOfMan/xdc1 · github.com/sunnyxxy/Star-Rating-Rebirth · github.com/LeoBlackMT/osumania_map_analyser · github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark · benchmark.leoblack.top · ieeexplore.ieee.org/document/11605932 · osu.ppy.sh/community/forums/topics/975109 · osu.ppy.sh/community/forums/topics/254145 · som1sezhi.github.io/itg-difficulty-predictor/

**Model growth / training / fusion:** arxiv.org/abs/1511.05641 · arxiv.org/abs/1705.07750 · arxiv.org/abs/2302.05543 · arxiv.org/abs/2003.04887 · arxiv.org/abs/2004.10964 · aclanthology.org/2020.acl-main.740/ · arxiv.org/abs/1801.06146 · arxiv.org/abs/2302.03241 · arxiv.org/abs/1706.02677 · arxiv.org/abs/2404.06395 · arxiv.org/abs/2106.09685 · arxiv.org/abs/2202.03555 · arxiv.org/abs/2106.11959 · arxiv.org/abs/1604.06737 · arxiv.org/abs/2203.05556 · arxiv.org/abs/1709.07871 · arxiv.org/abs/1612.08083 · arxiv.org/abs/1709.01507 · arxiv.org/abs/1807.03247

**Leakage/style:** github.com/OliBomby/Mapperatorinator/tree/main/classifier · dl.acm.org/doi/full/10.1145/3748336.3748345 · dl.acm.org/doi/10.1145/2382577.2382579 · scikit-learn.org/stable/common_pitfalls.html · doi.org/10.1007/s10844-013-0258-3

**Local:** data2vec_main/README.md · data2vec_main/d2v_main/charts.py · data2vec_main/configs/audio_style_base.yaml · dcr/cm3p_offline.py · docs/plans/2026-11-24_d2v_attr_evidence/code_touchpoints.md · docs/plans/2026-11-24_d2v_attr_evidence/proto_attr_roll.py · docs/plans/2026-11-24_behavior_supervision_plan.md (line 138)

---

## Appendix B — Evidence-strength caveats

- [P] means primary for the claim as stated, not peer-reviewed for our application: osu! wiki/RC and source code are authoritative for game mechanics, not difficulty-model ablations.
- Strongest transferable claims: representation choices (§1) and model-growth mechanics (§3), both peer-reviewed. Weakest: (b) attribute usefulness — only domain specs + our planned ablation.
- Two pages were unreachable from this sandbox at review time (MIREX tempo page; OpenReview is bot-protected); URLs retained from search results and labelled [S].
- Community sources (forums, ManiaMapAnalyser docs) labelled [S]; do not build the plan's central claim on them.

# Path 1-6 Failure Analysis

## 目的

这份文档用于总结当前 `multi_image_safety` 项目里 Path 1 到 Path 6 的真实运行状态、失败原因、跨路径共性问题，以及后续需要优先修改的方向。

它的目标不是介绍路径设计，而是回答两个问题：

1. 现在哪些 path 真正成功生成了最终可用数据？
2. 如果失败了，根本原因是什么，后续应该优先改哪里？

---

## 一句话结论

当前 **Path 1 到 Path 6 没有一条完整成功地产出最终可用样本**。

更具体地说：

1. `Path 2` 是目前最接近成功的一条，已经生成了较多中间结果，但最终验证没有完成。
2. `Path 3`、`Path 4`、`Path 5` 的主要问题是 **vLLM 初始化失败**。
3. `Path 1`、`Path 6` 的主要问题不是直接崩溃，而是 **过滤条件过严，候选被全部筛空**。

---

## vLLM 初始化失败的根本原因

## 结论

当前 `Path 3`、`Path 4`、`Path 5` 中反复出现的：

`RuntimeError: Engine core initialization failed`

**根本原因大概率是：Qwen/Qwen3.5-27B 在 4 卡启动时参数过于激进，导致 vLLM worker 在初始化显存阶段失败。**

更准确地说，是下面两个因素叠加：

1. `max_model_len=8192`
2. `gpu_memory_utilization=0.9`

这两个设置对于当前机器、当前剩余显存、以及当前启动方式来说太激进了。

---

## 为什么这样判断

### 证据 1：旧的 Path 2 日志已经暴露过底层真实错误

虽然很多日志最后只显示：

`Engine core initialization failed`

但在较早一次 Path 2 的失败日志中，已经能看到更底层的 worker 报错：

- `Free memory on device cuda:3 (33.49/47.54 GiB) on startup is less than desired GPU memory utilization (0.9, 42.78 GiB).`
- `Decrease GPU memory utilization or reduce GPU memory used by other processes.`

也就是说，`Engine core initialization failed` 在这个项目里并不是一个抽象错误，而是曾经已经明确展开过：

**本质上是 vLLM 启动时申请的显存预算超过了当时每张卡可用显存。**

相关证据见：

`logs/path2_20260408_170115.log`

---

### 证据 2：Path 3 / 4 / 5 仍然在用激进配置

当前代码里，`Path 3`、`Path 4`、`Path 5` 的 LLM 子流程仍然大量使用：

1. `max_model_len=8192`
2. `gpu_memory_utilization=0.9`

而这些正是之前被证明会导致 Qwen 27B 启动失败的组合。

典型位置包括：

1. `run_path3_expand.py`
   - Method A 子进程里：`max_model_len=8192`
   - `gpu_memory_utilization={local_cfg.get("gpu_memory_utilization", 0.9)}`
2. `src/path4_scenario/scene_gen.py`
   - `max_model_len=8192`
3. `src/path4_scenario/intent_inject.py`
   - `max_model_len=8192`
4. `run_path5_embedding.py`
   - Step 2 子进程里：`max_model_len=8192`
   - `gpu_memory_utilization={local_cfg.get("gpu_memory_utilization", 0.9)}`

相比之下，后来能正常初始化的路径已经切到了更保守的配置：

1. `max_model_len=4096`
2. `gpu_memory_utilization=0.68`
3. `disable_custom_all_reduce=True`

这套保守配置已经在 `Path 2`、`Path 1`、`Path 6` 的部分流程里证明更稳定。

---

### 证据 3：日志里还有一个“促发因素”

相关日志反复出现：

`We must use the spawn multiprocessing start method ... Reasons: CUDA is initialized`

这说明这些 path 的 vLLM 往往是在一个 **已经碰过 CUDA 的 Python 进程** 里再次拉起 worker。

这不是唯一根因，但它会让初始化更脆弱。

所以当前更准确的判断是：

1. **主因**：显存预算过高，27B 模型初始化参数太激进
2. **次因**：父进程已经初始化过 CUDA，导致 vLLM 被迫切换启动方式，进一步增加不稳定性

---

## Path 当前状态总表

| Path | 当前状态 | 是否有中间结果 | 是否有最终结果 |
|---|---|---|---|
| Path 1 | 失败 | 有 | 无 |
| Path 2 | 部分成功 | 有 | 无 |
| Path 3 | 失败 | 有 | 无 |
| Path 4 | 失败 | 很少/无 | 无 |
| Path 5 | 失败 | 有 | 无 |
| Path 6 | 失败 | 有 | 无 |

---

## Path 1：失败原因

## 当前状态

已经生成：

1. `numberbatch_pairs.jsonl`
2. `llm_pairs.jsonl`
3. `all_mined_pairs.jsonl`

但最终：

1. `filtered_pairs.jsonl = 0`
2. `pairs_with_images.jsonl = 0`
3. `validated_samples.jsonl = 0`

## 根因

Path 1 的主问题不是 vLLM 崩了，而是：

**CLIP 过滤阶段把所有候选概念对都筛掉了。**

日志里有非常明确的证据：

`CLIP filter: 0 / 1252 pairs passed`

相关日志：

`logs/path1_seq.log`

## 对应代码原因

`src/path1_kg_concept/pair_filter.py` 的过滤条件是：

1. `sim1 < theta_safe`
2. `sim2 < theta_safe`
3. `combined_sim > theta_harm`

默认阈值：

1. `theta_safe = 0.25`
2. `theta_harm = 0.35`

这意味着：

1. 单个概念必须足够“无害”
2. 组合后又必须足够“有害”

在当前概念生成质量和当前 harm vector 表示下，这个条件过于苛刻，所以 `1252` 个候选全部被清空。

## 结论

Path 1 当前是 **过滤策略过严**，不是主流程没跑起来。

---

## Path 2：部分成功，但最终未完成

## 当前状态

已经生成：

1. `collected_prompts.jsonl = 3801`
2. `decomposed_prompts.jsonl = 3659`
3. `samples_with_images.jsonl = 2041`

但最终：

1. `validated_samples.jsonl = 0`

## 根因

Path 2 当前不是单一原因，而是“历史上有多个问题叠加”：

1. 早期 Step 2 发生过 vLLM 初始化失败
2. Step 3 入口曾经缺失 `get_hf_token` 导致 `ImportError`
3. 当前 Step 3 状态还停留在 `.running`，说明最新一次运行没有正常完成 step state 收口
4. `validated_samples.jsonl` 的时间非常早，仍是旧的空文件，说明 Step 4 还没有真正产出新结果

## 直接证据

当前 Path 2 状态目录中存在：

1. `step1_collect_prompts.done.json`
2. `step2_decompose_prompts.done.json`
3. `step3_acquire_images.running.json`

但没有：

1. `step3_acquire_images.done.json`
2. `step4_validate_samples.done.json`

说明当前 Path 2 的最新状态是：

**Step 1 和 Step 2 完成了，Step 3 曾经在进行中，但没有被正常收口；Step 4 实际上没有完成。**

## 结论

Path 2 是目前最接近成功的一条，但还不能算成功。
它已经有大量可用中间结果，但最终样本文件还没真正生成出来。

---

## Path 3：失败原因

## 当前状态

已经生成：

1. `all_image_infos.jsonl = 4076`

但最终：

1. `cross_paired_samples.jsonl = 0`

## 根因

Path 3 有两类问题：

### 主问题：Method A 和 Method B 都卡在 vLLM 初始化

日志中两个子流程都报：

`RuntimeError: Engine core initialization failed`

而且两边都发生在 Qwen 启动阶段。

### 次问题：外部数据源和依赖不完整

日志里还能看到：

1. `Failed to download VLGuard`
2. `Failed to download BeaverTails-V: Config name is missing`
3. `MIS train JSON not found at /mnt/hdd/xuran/MIS/mis_train/mis_train.json`
4. 多次 `Openverse retrieval failed: HTTP Error 403: Forbidden`

这些问题虽然不是最终归零的唯一原因，但会降低 Path 3 的数据覆盖度和可用输入质量。

## 为什么最终是 0

因为 Method A 和 Method B 都挂了：

1. Method A subprocess failed
2. Method B subprocess failed

最后脚本只是“继续执行并写出一个空结果文件”，不是成功产出了样本。

## 结论

Path 3 当前的主阻塞点是 **vLLM 初始化失败**，外加若干数据源问题。

---

## Path 4：失败原因

## 当前状态

Path 4 没有成功生成最终样本。

## 根因

Path 4 在最开始的 Step 1 就挂了：

1. `PATH 4 STEP 1: LLM scene generation`
2. `RuntimeError: Engine core initialization failed`
3. 随后 `RuntimeError: Step 1 (scene_gen) failed`

也就是说：

**Path 4 根本没有进入后面的 intent injection 和图像获取阶段。**

## 对应代码原因

`src/path4_scenario/scene_gen.py` 里当前仍是：

1. `max_model_len=8192`
2. 无保守显存策略

而 `intent_inject.py` 也是相同风格。

## 结论

Path 4 是典型的 **第一步就被 vLLM 初始化卡死**。

---

## Path 5：失败原因

## 当前状态

已经生成：

1. `crawled_image_info.jsonl = 78`
2. `_cross_pair_input.jsonl = 1527`

但最终没有生成 prompt 增强后的完整结果。

## 根因

Path 5 的问题分成两层：

### 第一层：外部图像检索效果较差

日志中有多次：

`Openverse retrieval failed: HTTP Error 403: Forbidden`

这导致 Step 1 最终只拿到了 `78` 张图，图像池偏小。

### 第二层：Step 2 的 vLLM cross-pairing 启动失败

在真正关键的 Step 2 中，日志再次出现：

`RuntimeError: Engine core initialization failed`

所以虽然候选对已经构造出来了，但 LLM 没能完成最终判断和文本生成。

## 结论

Path 5 的主问题依然是 **vLLM 初始化失败**；
外部检索 403 是次要但真实存在的问题。

---

## Path 6：失败原因

## 当前状态

已经生成：

1. `raw_chains.jsonl = 257`

但之后：

1. `scored_chains.jsonl = 0`
2. `fusion_pairs.jsonl = 0`
3. `pairs_with_images.jsonl = 0`
4. `validated_samples.jsonl = 0`

## 根因

Path 6 不是先崩在 vLLM，而是：

**链生成成功了，但 CLIP 链评分把全部链都筛掉了。**

日志里有明确证据：

1. `Total chains generated: 257`
2. `Chain CLIP scoring: 0 / 257 chains passed`

## 对应代码原因

`src/path6_tag_kg_fusion/tag_builder.py` 的过滤逻辑要求：

1. 链首概念和链尾概念都必须足够无害
2. 首尾概念组合后又必须足够接近 harm vector

默认阈值：

1. `theta_safe = 0.25`
2. `theta_harm = 0.30`

这与 Path 1 的问题非常类似：

**约束太强，导致生成出的链全部在评分阶段被筛空。**

## 次级依赖问题

Path 6 Step 3 还依赖 Path 1 的产物做 fusion。
而 Path 1 当前本身就是空的。

所以即使 Path 6 的链评分不过空，后续融合阶段也仍然会受到 Path 1 失败的影响。

## 结论

Path 6 当前的主要问题是 **链评分阈值过严**，并且还受到 Path 1 失败的连带影响。

---

## 当前跨路径共性问题

## 问题 1：vLLM 参数不统一

现在项目里存在两套思路：

1. Path 2 / Path 1 / Path 6 的部分流程已经切到保守配置
2. Path 3 / Path 4 / Path 5 仍然大量保留 `8192 + 0.9`

这会导致：

1. 有的路径能初始化
2. 有的路径一直死在 worker 启动

这是当前最需要统一的地方。

---

## 问题 2：父进程已初始化 CUDA

很多日志里都能看到：

`We must use the spawn multiprocessing start method ... Reasons: CUDA is initialized`

这说明不少路径是在“已经碰过 CUDA 的进程”里再去拉 vLLM worker。

这会增加初始化不稳定性，尤其在大模型多卡情况下更明显。

---

## 问题 3：过滤阈值整体偏严

Path 1 和 Path 6 当前都出现了同类问题：

1. 前置生成成功
2. 后置 CLIP 过滤全部归零

这说明当前：

1. 候选生成质量不够好
2. 或 CLIP harm vector / threshold 设置不适配
3. 或两者同时存在

---

## 问题 4：外部数据源不稳定

当前已观测到：

1. `Openverse 403`
2. `VLGuard` 下载失败
3. `BeaverTails-V` config 问题
4. 本地 `MIS train JSON` 缺失

这会直接影响 Path 3 和 Path 5 的输入质量与覆盖范围。

---

## 问题 5：缓存路径迁移尚未真正完成

项目代码里已经有一部分默认路径被改向 `/mnt2/xuran_hdd/cache`，
但系统层面 `/mnt2` 一度处于只读挂载，真实缓存仍在旧路径。

这意味着后续还存在潜在风险：

1. 代码默认指向新路径
2. 真实缓存还在旧路径
3. 如果环境变量和软链没有统一，后续模型加载可能再次出现缓存分裂或找不到 token 的问题

---

## 后续建议：优先修改顺序

如果后续要让另一个 AI 来继续修改，建议按这个顺序：

### 第一优先级

统一修复 `Path 3 / Path 4 / Path 5` 的 vLLM 启动配置：

1. 把 `max_model_len=8192` 改成 `4096`
2. 把 `gpu_memory_utilization=0.9` 改成 `0.68`
3. 尽量统一加上 `disable_custom_all_reduce=True`
4. 尽量避免在已经初始化 CUDA 的进程里直接拉 vLLM worker

这是当前最值钱的一组修改，因为它能一次性影响三条主失败路径。

### 第二优先级

修复 Path 2 的 Step 3 / Step 4 收口问题：

1. 清理不完整的 `.step_state`
2. 确认 Step 3 能正确 finish
3. 跑通 Step 4 验证

因为 Path 2 已经最接近成功，优先把它收尾最划算。

### 第三优先级

放宽或重新设计 Path 1 / Path 6 的 CLIP 过滤策略：

1. 检查 harm vector 构造是否合适
2. 重新调 `theta_safe` 和 `theta_harm`
3. 先做采样分析，不要直接全量跑

### 第四优先级

补齐外部数据和检索稳定性：

1. 修 Openverse 403
2. 修 VLGuard / BeaverTails-V 下载配置
3. 补上本地 MIS 数据路径

---

## 最终判断

当前项目状态不是“完全不能跑”，而是：

1. **Path 2 基本已经半通**
2. **Path 3 / 4 / 5 卡在同一个核心问题：vLLM 初始化**
3. **Path 1 / 6 卡在另一个核心问题：过滤后全空**

所以后续最合理的策略不是逐条零散修，而是先做两类系统性修复：

1. 修 vLLM 启动参数和启动方式
2. 修 CLIP 过滤阈值和筛选逻辑

只要这两件事解决掉，整套 Path 1-6 的成功率会明显提升。

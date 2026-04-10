# 多图安全项目路径总览

## 项目目标

这个项目的目标是为 VLM / VLM 对齐研究构建一个多图安全数据集。
它的核心思想是：

1. 每一张单独的图片都应当看起来无害，或者最多只是轻微可疑。
2. 真正的有害意图只会在模型同时理解两张图和配套文本提示时出现。
3. 不同路径生成不同风格的组合式攻击样本，从而让最终数据集更加多样。

整个流程在概念上可以理解为：

1. 通过 Path 1 到 Path 6 生成候选样本。
2. 运行质量控制和过滤。
3. 将保留下来的样本导出为最终训练格式。

---

## 共同模式

大多数路径都遵循类似的高层流程：

1. 构造候选概念、提示词、场景或图像池。
2. 用 LLM 对这些内容进行拆分、配对、融合或重写，生成隐蔽的双图攻击样本。
3. 通过文生图、检索，或两者结合获取真实图像。
4. 验证单图安全性和组合后有害性。
5. 将每条路径的结果写入 `data/raw/pathX/`。

主要启动脚本如下：

1. `scripts/run_path1.sh`
2. `scripts/run_path2.sh`
3. `scripts/run_path3.sh`
4. `scripts/run_path4.sh`
5. `scripts/run_path5.sh`
6. `scripts/run_path6.sh`

---

## Path 1：KG 概念对挖掘

### 核心思路

Path 1 不是从用户提示词出发，而是从概念对出发。
它的目标是找到两个单独看都很无害的概念，但当模型把它们联系起来时，会产生有害含义。
这一路径特别适合构造隐蔽、间接、依赖联想推理的攻击样本。

### 详细步骤

1. 从结构化或半结构化知识源中挖掘概念对。
   一条分支使用 Numberbatch 风格的语义邻居。
   另一条分支使用 LLM 直接生成候选概念对。
2. 将所有挖掘出的概念对合并成一个候选集合。
3. 使用基于 CLIP 的过滤，保留那些单独看起来安全、但组合后具有有害暗示的概念对。
4. 为概念对中的两个概念分别生成无害图片。
5. 使用 LLM 生成连接性文本提示，使这两个概念只有在组合时才体现有害性。

### 主要入口

`run_path1_kg_concept.py`

### 关键输出

1. `data/raw/path1/numberbatch_pairs.jsonl`
2. `data/raw/path1/llm_pairs.jsonl`
3. `data/raw/path1/filtered_pairs.jsonl`
4. `data/raw/path1/pairs_with_images.jsonl`
5. `data/raw/path1/validated_samples.jsonl`

---

## Path 2：提示词拆分

### 核心思路

Path 2 从显式有害的文本提示词出发。
它会让 LLM 将一个有害请求拆成两个图像描述，使得两张图单独看起来都无害。
这是最典型、最直接的“把有害意图拆散到两张图里”的路径。

### 详细步骤

1. 从多个文本安全数据集中收集有害提示词。
   当前代码使用 BeaverTails、SorryBench 和 ToxicChat。
2. 将这些提示词映射或归一化到项目内部的 taxonomy。
3. 使用 LLM 将每个有害提示词拆分成：
   `image1_description`、`image2_description` 和连接它们的 `text_prompt`。
4. 为两个描述获取图像。
   默认优先生成，如果失败再尝试检索。
5. 对生成的样本进行单图安全和组合有害性验证。

### 主要入口

`run_path2.py`

### 相关模块

1. `src/path2_prompt_decompose/collect_prompts.py`
2. `src/path2_prompt_decompose/decompose.py`
3. `src/path2_prompt_decompose/acquire_images.py`
4. `src/path2_prompt_decompose/validate.py`

### 关键输出

1. `data/raw/path2/collected_prompts.jsonl`
2. `data/raw/path2/decomposed_prompts.jsonl`
3. `data/raw/path2/samples_with_images.jsonl`
4. `data/raw/path2/validated_samples.jsonl`

---

## Path 3：数据集扩展

### 核心思路

Path 3 不是从零开始生成，而是从现有的安全相关图像数据集出发进行扩展。
它的目标是把已有的带描述或带标签图像，重新组织成新的双图组合攻击样本。

### 详细步骤

1. 从 Hugging Face 安全数据集和可选的外部检索源中收集图像与元数据。
2. Method A：
   对带 caption 或文本描述的图像进行拆分，构造文本级的双图攻击结构。
   这个阶段先产出文本样本，真实图片可以之后再补。
3. Method B：
   直接从已有图像池中构造跨图像配对。
   然后让 LLM 判断哪些图像对在组合后会产生有害含义，并生成连接文本提示。
4. 将 Method A 和 Method B 合并成当前 Path 3 的主输出。
5. 如果 Method A 还需要真实图片，可以再额外运行 `run_path3_acquire_images.py`。

### 主要入口

`run_path3_expand.py`

### 可选后续步骤

`run_path3_acquire_images.py`

### 关键输出

1. `data/raw/path3/all_image_infos.jsonl`
2. `data/raw/path3/cross_paired_samples.jsonl`
3. `data/raw/path3/STATUS.md`

### 额外说明

Path 3 是最混合的一条路径。
在可选的补图步骤之前，一部分输出可能仍然只有文本，而另一部分已经自带真实图像路径。

---

## Path 4：场景构造

### 核心思路

Path 4 从日常场景和普通活动出发。
它先构造正常生活化场景，再向其中注入有害意图，使图片本身仍然普通，但组合解释会变得不安全。
这条路径特别适合生成更真实、更上下文化的攻击样本。

### 详细步骤

1. 使用 LLM 生成日常场景和活动对。
2. 再通过另一轮 LLM，将危险意图注入这些原本正常的场景中。
3. 获取对应图像。
   当前实现支持 LAION 风格检索，以及 T2I 作为补充。
4. 保存得到的场景型多图样本。

### 主要入口

`run_path4_scenario.py`

### 关键输出

具体文件名会随着当前实现细节略有变化，但最终目标是产出一批带图像路径和注入意图的 Path 4 样本。

---

## Path 5：图像对挖掘

### 核心思路

Path 5 从图像池出发，而不是从文本提示词出发。
它会先按有害类别检索一批“表面无害但语义相关”的图片，再让 LLM 判断哪些图像对在组合时会产生有害解读。
因此，这是一条“挖掘型”路径，而不是“生成优先”路径。

### 详细步骤

1. 从外部源中检索或抓取各个有害类别对应的图像。
2. 按类别组织这些图像，并构造候选图像对。
3. 使用 LLM 判断每个图像对是否能够在组合后形成隐蔽的有害语义。
4. 对于通过的图像对，生成连接两张图的文本提示。

### 主要入口

`run_path5_embedding.py`

### 关键输出

1. `data/raw/path5/crawled_image_info.jsonl`
2. `data/raw/path5/_cross_pair_input.jsonl`
3. 脚本最终产出的 Path 5 文本提示增强结果文件

---

## Path 6：TAG + KG 融合

### 核心思路

Path 6 是结构性最强、隐蔽性最高的一条路径。
它先构造 toxicity association chain，再进行过滤，并与 Path 1 的 KG 概念对融合。
目标是构造一种多跳推理式的攻击样本，让有害意图隐藏在概念链而非直接提示词中。

### 详细步骤

1. 使用 LLM 生成 TAG 风格的概念链。
2. 使用 CLIP 对这些链打分，保留那些足够隐蔽但又有效的链。
3. 从保留下来的链中提取端点概念对。
4. 如果有 Path 1 的 KG 对，则将其与这些端点对进行融合。
5. 为融合后的概念对生成图片。
6. 使用 LLM 生成连接性的文本提示。
7. 对生成结果进行 MTC 或隐蔽性打分。

### 主要入口

`run_path6_tag_kg.py`

### 关键输出

1. `data/raw/path6/raw_chains.jsonl`
2. `data/raw/path6/scored_chains.jsonl`
3. `data/raw/path6/fusion_pairs.jsonl`
4. `data/raw/path6/pairs_with_images.jsonl`
5. `data/raw/path6/validated_samples.jsonl`

---

## 各路径之间的差异

1. Path 1 是 concept-first，偏图谱 / 语义联想驱动。
2. Path 2 是 prompt-first，偏拆分驱动。
3. Path 3 是 dataset-first，偏复用和扩展驱动。
4. Path 4 是 scene-first，偏场景与上下文注入驱动。
5. Path 5 是 retrieval-first，偏图像对挖掘驱动。
6. Path 6 是 chain-first，偏多跳链式融合驱动。

它们组合起来，覆盖了直接拆分、语义联想、场景操控、检索配对，以及结构化图谱融合等多种攻击构造方式。

---

## 建议的代码阅读顺序

如果你想最快理解整个项目，建议按下面顺序阅读：

1. `src/pipeline/run_path.py`
2. `run_path2.py`
3. `run_path1_kg_concept.py`
4. `run_path3_expand.py`
5. `run_path4_scenario.py`
6. `run_path5_embedding.py`
7. `run_path6_tag_kg.py`

然后再看这些公共模块：

1. `src/common/image_generation.py`
2. `src/common/clip_utils.py`
3. `src/common/schema.py`
4. `config/pipeline.yaml`
5. `config/taxonomy.yaml`

---

## 所有路径之后的最终阶段

当各条路径都生成完候选样本后，项目还需要一个最终质量控制阶段：

1. 合并所有路径的输出
2. 单图安全检查
3. 组合有害性验证
4. 去重
5. 分布平衡控制
6. 导出为最终训练格式

这个最终阶段才会把六条路径分别生成的候选池，真正转化成最后的数据集发布版本。

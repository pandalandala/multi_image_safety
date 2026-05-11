# 多图安全项目路径产物总览

这份文档基于 **2026-05-04 当前磁盘上的真实产物** 重写，重点回答两个问题：

1. 每个 Path 现在的 sample 到底落在哪个 `jsonl` 文件里。
2. 每个 sample 是否都能找到对应的 `text_prompt`、`image1_path`、`image2_path`。

这份文档适合在后续把 6 个 Path 合并成一个统一数据集时使用。

## 先看结论

建议作为“可直接并入总数据集”的最终 sample 文件：

| Path | 建议使用的最终文件 | 记录数 | 有 `text_prompt` | 两张图路径都存在 | 现成 sample id |
| --- | --- | ---: | --- | --- | --- |
| Path 1 | `data/raw/path1/validated_samples.jsonl` | 1221 | 是 | 是 | 否 |
| Path 2 | `data/raw/path2/validated_samples.jsonl` | 3595 | 是 | 是 | `sample_id` |
| Path 3 | `data/raw/path3/cross_paired_samples.jsonl` | 7756 | 是 | 是 | `sample_id`, `sample_id_global` |
| Path 4 | `data/raw/path4/samples_with_images.jsonl` | 2650 | 是 | 是 | `sample_id` |
| Path 5 | `data/raw/path5/final_samples.jsonl` | 5000 | 是 | 是 | 否 |
| Path 6 | `data/raw/path6/validated_samples.jsonl` | 1081 | 是 | 是 | 否 |

如果你只是要合并六条路径的“最终可训练样本”，优先读上面这 6 个文件就够了。

## 合并时最需要注意的事

1. 不是每条 Path 的最终文件都有现成的稳定 `sample_id`。
2. `Path 1`、`Path 5`、`Path 6` 当前最终文件里都没有统一的 sample id，合并时建议你自己补一个全局键，例如 `merge_id = "{source_path}:{row_idx}"`。
3. 有些路径存在“上游 prompt-only 文件”或“有图但还没生成 prompt 的中间文件”，不要误把它们当最终样本。
4. `step_state` 有少量残留脏标记，尤其是 `Path 1` 和 `Path 4`，所以**以最终 jsonl 文件是否存在、字段是否完整为准**，不要只看 `.step_state`。

## 推荐统一字段

如果后面要把 6 个 Path 拼成一个总表，建议至少统一成下面这些字段：

1. `merge_id`
2. `source_path`
3. `path_name`
4. `category`
5. `text_prompt`
6. `image1_path`
7. `image2_path`
8. `image1_description`
9. `image2_description`
10. `reasoning`
11. `safety_response` 可选

---

## Path 1

### 当前建议使用的最终文件

- `data/raw/path1/validated_samples.jsonl`

### 当前真实状态

- 记录数：`1221`
- `text_prompt`：`1221/1221`
- `image1_path` 存在：`1221/1221`
- `image2_path` 存在：`1221/1221`
- 现成 sample id：**没有**

### 相关文件怎么区分

- `data/raw/path1/validated_samples.jsonl`
  最终文件。带两张图、`text_prompt`、`reasoning`、`safety_response`。
- `data/raw/path1/pairs_with_images.jsonl`
  中间文件。`1652` 条，图都有，但**没有** `text_prompt`，不能直接当最终样本合并。

### 字段说明

最终文件里主要有：

- `category`
- `concept1`, `concept2`
- `image1_path`, `image2_path`
- `image1_description`, `image2_description`
- `text_prompt`
- `reasoning`
- `safety_response`
- `source_path`

### 特别提醒

- `Path 1` 的 `.step_state` 目前不干净：
  - 有 `step3_generate_images.done.json`
  - 有 `step4_generate_prompts.done.json`
  - 但也残留 `step1a_numberbatch_pairs.running.json`
  - 以及 `step2_filter_pairs.failed.json`
- 因此这里应当以 `validated_samples.jsonl` 为准，而不是以 step 标记为准。

---

## Path 2

### 当前建议使用的最终文件

- `data/raw/path2/validated_samples.jsonl`

### 当前真实状态

- 记录数：`3595`
- `text_prompt`：`3595/3595`
- `image1_path` 存在：`3595/3595`
- `image2_path` 存在：`3595/3595`
- 现成 sample id：`sample_id`

### 相关文件怎么区分

- `data/raw/path2/validated_samples.jsonl`
  最终文件。已经过验证，适合直接并入总数据集。
- `data/raw/path2/samples_with_images.jsonl`
  上一步中间文件。`3615` 条，图和 prompt 都有，但还没过最终验证。
- `data/raw/path2/decomposed_prompts.jsonl`
  prompt-only 中间文件。`3677` 条，有 `text_prompt` 和两张图描述，但没有图路径。
- `data/raw/path2/collected_prompts.jsonl`
  最原始文本池。`3821` 条，不是多图 sample。

### 字段说明

最终文件里主要有：

- `sample_id`
- `category`
- `text`
- `text_prompt`
- `image1_path`, `image2_path`
- `image1_description`, `image2_description`
- `reasoning`
- `individual_safety_scores`
- `source_path`

### Step 状态

- `step1_collect_prompts.done.json`
- `step2_decompose_prompts.done.json`
- `step3_acquire_images.done.json`
- `step4_validate_samples.done.json`

这条路径目前是干净完成状态。

---

## Path 3

### 当前建议使用的最终文件

- `data/raw/path3/cross_paired_samples.jsonl`

### 当前真实状态

- 记录数：`7756`
- `text_prompt`：`7756/7756`
- `image1_path` 存在：`7756/7756`
- `image2_path` 存在：`7756/7756`
- 现成 sample id：`sample_id`, `sample_id_global`

### 相关文件怎么区分

- `data/raw/path3/cross_paired_samples.jsonl`
  最终合并文件，等于 Method A 和 Method B 的并集。
- `data/raw/path3/method_a_with_images.jsonl`
  `6825` 条。Method A 分支的最终可用样本，有图、有 prompt。
- `data/raw/path3/method_b_cross_paired.jsonl`
  `931` 条。Method B 分支的最终可用样本，有图、有 prompt。
- `data/raw/path3/method_a_decomposed.jsonl`
  `6983` 条。Method A 的 prompt-only 文件，有描述和 `text_prompt`，但没有图路径。

### 路径内部的关系

- `cross_paired_samples.jsonl = method_a_with_images.jsonl + method_b_cross_paired.jsonl`
- `6983` 条 Method A 文本样本里，最终只有 `6825` 条成功补到图
- 所以如果你只要最终可用样本，直接使用 `cross_paired_samples.jsonl`

### 字段说明

最终文件里主要有：

- `sample_id`
- `sample_id_global`
- `category`
- `text_prompt`
- `image1_path`, `image2_path`
- `image1_description`, `image2_description`
- `reasoning`
- `source_dataset`
- `source_path`

### Step 状态

- `step1_collect_image_pool.done.json`
- `step2_method_a_decompose.done.json`
- `step3_method_b_cross_pair.done.json`
- `step4_merge_output.done.json`
- `step5_method_a_acquire_images.done.json`

这条路径目前是完整完成状态。

---

## Path 4

### 当前建议使用的最终文件

- `data/raw/path4/samples_with_images.jsonl`

### 当前真实状态

- 记录数：`2650`
- `text_prompt`：`2650/2650`
- `image1_path` 存在：`2650/2650`
- `image2_path` 存在：`2650/2650`
- 现成 sample id：`sample_id`

### 相关文件怎么区分

- `data/raw/path4/samples_with_images.jsonl`
  最终可用文件，有 prompt，也有两张图。
- `data/raw/path4/intent_injected_samples.jsonl`
  `2721` 条。上游 prompt-only 文件，有 `text_prompt` 和图描述，但没有图路径。

### 路径内部的关系

- 上游共生成了 `2721` 条 intent-injected samples
- 最终成功补图的是 `2650` 条
- 也就是说这条路径目前有 `71` 条 prompt-only 样本没有进入最终带图文件

### 字段说明

最终文件里主要有：

- `sample_id`
- `category`
- `scene_category`
- `activity`
- `safety_mode`
- `text_prompt`
- `image1_path`, `image2_path`
- `image1_description`, `image2_description`
- `reasoning`
- `source_path`

### 特别提醒

- `Path 4` 当前存在残留的 `step1_generate_scenes.running.json`
- 但 `step3_fetch_images.done.json` 和最终 `samples_with_images.jsonl` 都已经存在
- 所以和 `Path 1` 一样，这里也应当以最终文件为准，而不是只看 step-state

---

## Path 5

### 当前建议使用的最终文件

- `data/raw/path5/final_samples.jsonl`

### 当前真实状态

- 记录数：`5000`
- `text_prompt`：`5000/5000`
- `image1_path` 存在：`5000/5000`
- `image2_path` 存在：`5000/5000`
- 现成 sample id：**没有**

### 相关文件怎么区分

- `data/raw/path5/final_samples.jsonl`
  当前建议直接用于汇总的最终文件。
- `data/raw/path5/samples_with_prompts.jsonl`
  `18403` 条。更大的已接受样本池，也有 prompt 和两张图，但还没裁成最终规模。
- `data/raw/path5/crawled_image_info.jsonl`
  `889` 条。这是单图 crawl 池，不是双图 sample 文件。

### 路径内部的关系

- `samples_with_prompts.jsonl` 是更大的候选池
- `final_samples.jsonl` 是下游更适合直接使用的固定规模版本

### 字段说明

最终文件里主要有：

- `category`
- `text_prompt`
- `image1_path`, `image2_path`
- `image1_description`, `image2_description`
- `image1_query`, `image2_query`
- `image1_class`, `image2_class`
- `reasoning`
- `confidence`
- `source_path`

### Step 状态

- `step1_acquire_images.done.json`
- `step2_cross_pair_prompts.done.json`

这条路径的最终文件已经存在并可用。

---

## Path 6

### 当前建议使用的最终文件

- `data/raw/path6/validated_samples.jsonl`

### 当前真实状态

- 记录数：`1081`
- `text_prompt`：`1081/1081`
- `image1_path` 存在：`1081/1081`
- `image2_path` 存在：`1081/1081`
- 现成 sample id：**没有**

### 相关文件怎么区分

- `data/raw/path6/validated_samples.jsonl`
  最终文件。带图、带 prompt、带 `safety_response`。
- `data/raw/path6/pairs_with_images.jsonl`
  `1504` 条。图都有，但没有 `text_prompt`。
- `data/raw/path6/fusion_pairs.jsonl`
  `1787` 条。只有概念 pair，没有图，也没有 prompt。

### 路径内部的关系

- `fusion_pairs.jsonl` 是概念层中间结果
- `pairs_with_images.jsonl` 是已补图但未生成 prompt 的中间结果
- `validated_samples.jsonl` 才是最终可直接并入总数据集的版本

### 字段说明

最终文件里主要有：

- `category`
- `sub_category`
- `concept1`, `concept2`
- `full_chain`
- `hop_count`
- `fusion_source`
- `text_prompt`
- `image1_path`, `image2_path`
- `reasoning`
- `safety_response`
- `source_path`

### Step 状态

- `step1_generate_chains.done.json`
- `step2_score_chains.done.json`
- `step3_fuse_pairs.done.json`
- `step4_generate_images.done.json`
- `step5_generate_prompts.done.json`

这条路径目前是完整完成状态。

---

## 汇总时的建议用法

如果你的目标是把六条路径的产物合成一个统一总集，建议优先读取下面这 6 个文件：

1. `data/raw/path1/validated_samples.jsonl`
2. `data/raw/path2/validated_samples.jsonl`
3. `data/raw/path3/cross_paired_samples.jsonl`
4. `data/raw/path4/samples_with_images.jsonl`
5. `data/raw/path5/final_samples.jsonl`
6. `data/raw/path6/validated_samples.jsonl`

这些文件当前都满足：

1. 每条样本都能找到 `text_prompt`
2. 每条样本都有 `image1_path`
3. 每条样本都有 `image2_path`
4. 路径指向的图片文件当前都存在于磁盘上

## 不建议直接并入的文件

下面这些文件虽然有用，但不建议直接作为最终样本汇总源：

1. `path1/pairs_with_images.jsonl`
   有图，但没有 prompt。
2. `path2/decomposed_prompts.jsonl`
   有 prompt，但没有图。
3. `path3/method_a_decomposed.jsonl`
   有 prompt，但没有图。
4. `path4/intent_injected_samples.jsonl`
   有 prompt，但没有图。
5. `path5/crawled_image_info.jsonl`
   是单图池，不是 sample。
6. `path6/fusion_pairs.jsonl`
   是概念 pair，不是最终多图 sample。
7. `path6/pairs_with_images.jsonl`
   有图，但没有 prompt。

## 合并前的最后一个提醒

合并时最容易踩坑的不是“图找不到”，而是：

1. 不同路径对 sample id 的支持不一致
2. 中间文件和最终文件名字很像
3. `reasoning` 字段在各路径都有，但含义偏“构造理由”，不是统一定义的标注答案
4. `safety_response` 目前只在部分路径中存在，不应当假设所有路径都有

如果后面真要落一个统一合并脚本，建议第一步先给所有最终文件补齐：

1. `path_name`
2. `merge_id`
3. 统一后的 `category`
4. 统一后的 `image1_path`, `image2_path`, `text_prompt`

这样后面的训练导出会稳很多。

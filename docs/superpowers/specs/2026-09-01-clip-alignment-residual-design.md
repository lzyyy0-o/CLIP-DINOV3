# CLIP 多尺度对齐与原始相关性残差设计

## 目标

提升 `clip_guided_shared_lite_vit` 在 Houston 2013 已见类训练、未见类测试协议下的开放词汇分割能力。当前训练只使用已见类的掩码交叉熵；虽然 CLIP 图像特征和文本原型参与了相关性解码，但没有目标直接约束 HSI--LiDAR 联合表示靠近 CLIP 的视觉语义空间。新增的设计必须保留现有四尺度文本相关性解码器、类别共享分类头和训练/测试词表切换协议。

本次范围仅覆盖 CLIP 引导架构；旧的 `teacher_student` 架构、数据划分和评测定义不改变。

## 决策

采用两个互补部件：

1. **多尺度 CLIP 对齐损失（训练时）**。用冻结的 CLIP 图像金字塔作为教师目标，使 Shared Lite-ViT 的 HSI--LiDAR 联合金字塔在每一个尺度学习相同的 512 维语义方向。
2. **原始 CLIP 文本相关性残差（推理和训练时）**。将四个尺度的 CLIP 图像--文本余弦相关图上采样并求平均，以固定正系数直接加到学习型相关性解码器的 logits 上。

选择这一组合而不是仅增加训练轮数或解冻 CLIP：前者不能消除已见类偏置，后者会在小样本 HSI--LiDAR 数据上损害 CLIP 已有的视觉--文本对齐。固定残差还保证训练后的类别共享 FPN 不会完全抹去 CLIP 对新文本类别的原始响应。

## 数据流

```text
HSI + LiDAR -> Shared Lite-ViT -> MMFB + CMFEB -> joint pyramid (4 x 512)
                                                     | \
                                                     |  \-- cosine alignment, target is detached
pseudo RGB -> frozen CLIP image tower -> clip pyramid (4 x 512)  |
                                                                    v
class names -> frozen CLIP text tower -> prompt text prototypes -> correlations
                                                         |                 |
joint/text correlations + CLIP/text correlations -> text-guided cost decoder
four-scale raw CLIP/text correlation average --------> fixed residual
                                                         |
                                                     full-vocabulary logits
```

The train loader still supplies a seen-class supervision mask. Cross-entropy is evaluated only where that mask corresponds to a configured seen class. The alignment loss uses that same valid training-pixel mask, resized nearest-neighbour at each pyramid level; therefore it does not add test-label supervision or make the protocol transductive.

## Model interface changes

`ClipGuidedSegmentationOutput` will retain `logits` and add:

- `joint_features: FeaturePyramid` -- four projected HSI--LiDAR feature maps;
- `clip_features: FeaturePyramid` -- four CLIP visual feature maps used by the decoder.

The segmentor already computes both pyramids. Returning them avoids a second CLIP forward pass and makes the loss inspectable. The trainer API remains `objective(output, labels, valid_mask)`.

## Multi-scale alignment objective

Add `ClipGuidedAlignmentObjective`, derived from the existing masked cross-entropy behaviour.

- Segmentation term: unchanged masked multiclass cross-entropy over configured seen classes.
- At each of the four aligned levels, L2-normalize the joint and CLIP feature vectors along channels, calculate `1 - cosine_similarity`, then average only valid resized training pixels.
- Detach the CLIP target before the comparison. Thus the loss updates the HSI/LiDAR branch, MMFB, CMFEB and joint projector, but never the CLIP image tower or its feature projections through this loss.
- Average the four scale losses into `clip_alignment`.
- Total loss is `segmentation + clip_alignment_weight * clip_alignment`.

The default experiment sets `clip_alignment_weight: 0.1`. This is intentionally moderate: the segmentation term must still fit the labelled seen classes, while the direction-level alignment continually preserves a text-compatible representation. Batch size one is supported because the loss is pixelwise and contains no batch-negative dependence.

`loss.kind` gains `clip_guided_alignment`. The CLIP-guided architecture accepts either the legacy `masked_cross_entropy` baseline or this new loss. The new kind requires a positive alignment weight and continues to require all legacy teacher-student loss weights to be zero. The Houston configuration will opt into the new kind, preserving the old mode for ablations.

## Raw CLIP correlation residual

`TextCorrelationDecoder` already evaluates a CLIP feature map against every prompt prototype at each pyramid scale. It will additionally retain those four raw maps before they enter learned embeddings or aggregators:

1. Resize every `[batch, class, height, width]` raw CLIP correlation map to the requested output size using bilinear interpolation.
2. Average the four resized maps.
3. Add `0.25 * averaged_raw_clip_correlation` to the learned FPN logits.

The coefficient is a fixed, positive decoder constant for this first experiment. It is not trainable, so optimization cannot set it to zero or reverse it. Correlation maps use normalized CLIP features and normalized text prototypes, so their scale is bounded and compatible with a conservative residual. The learned path remains responsible for modality-specific spatial refinement; the residual supplies a persistent CLIP zero-shot semantic prior.

## Configuration and compatibility

The Houston YAML changes only its loss selection and alignment coefficient:

```yaml
loss:
  kind: clip_guided_alignment
  clip_alignment_weight: 0.1
```

All model dimensions, six online ViT blocks / four emitted stages, frozen CLIP setting (`unfreeze_blocks: 0`), prompt templates, seen/unseen IDs, tile size, optimizer and early-stopping settings remain unchanged. Existing configurations that select `masked_cross_entropy` must continue to parse and train with no alignment term.

## Validation and tests

Tests will verify:

1. the model output exposes shape-consistent four-level joint and CLIP pyramids;
2. identical normalized teacher/student features produce approximately zero alignment loss, altered joint features increase it, and no gradient reaches the CLIP target through alignment;
3. alignment obeys the valid-pixel mask at each scale;
4. the decoder still emits non-zero class-dependent logits from raw CLIP correlations when the learnable decoding path is zeroed;
5. parser validation rejects non-positive alignment weights and permits the legacy baseline;
6. the full unit suite, Ruff, and Houston configuration validation pass.

## Non-goals and risks

This change does not claim that frozen CLIP alone guarantees unseen-class IoU. It improves the information path required for zero-shot transfer, but success still depends on semantically useful pseudo-RGB construction, prompt quality, the spatial split, and whether unseen Houston labels have visually related CLIP concepts. It also does not add unseen labels to training, change dataset splits, or unfreeze the CLIP towers.

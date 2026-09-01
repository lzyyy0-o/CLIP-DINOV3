from __future__ import annotations

import torch

from hsi_lidar_ovseg.vocabulary import ClassVocabulary


def test_vocabulary_decodes_local_logit_indices_to_global_class_ids() -> None:
    vocabulary = ClassVocabulary(class_ids=(2, 5), class_names=("trees", "road"))
    logits = torch.tensor([[[1.0, 4.0]], [[5.0, 0.0]]])

    predictions = vocabulary.decode_logits(logits)

    assert predictions.tolist() == [[5, 2]]


def test_vocabulary_counts_predictions_in_global_class_id_space() -> None:
    vocabulary = ClassVocabulary(class_ids=(2, 5), class_names=("trees", "road"))

    counts = vocabulary.prediction_counts(torch.tensor([[5, 2, 5]]))

    assert counts == {2: 1, 5: 2}

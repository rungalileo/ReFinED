from typing import Optional, Tuple

import torch
from torch import nn
from torch import Tensor
import dataquality as dq

from refined.doc_preprocessing.preprocessor import Preprocessor, PreprocessorInferenceOnly


class EntityTyping(nn.Module):
    def __init__(
        self,
        dropout: float,
        num_classes: int,
        encoder_hidden_size: int,
        preprocessor: Optional[PreprocessorInferenceOnly] = None
    ):
        super().__init__()
        # 🔭🌕 Galileo
        self.preprocessor = preprocessor
        self.dropout = nn.Dropout(dropout)  # not needed
        self.linear = nn.Linear(encoder_hidden_size, num_classes)
        self.init_weights()

    def init_weights(self):
        """Initialize weights for all member variables with type nn.Module"""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(
        self, mention_embeddings: Tensor, span_classes: Optional[Tensor] = None, entity_ids: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Tensor]:
        """
        Forward pass of Entity Typing layer.
        Averages the encoder's contextualised token embeddings for each mention span.
        :param mention_embeddings: mention embeddings
        :param span_classes: class targets
        :return: loss tensor (if span_classes is provided), class activations tensor
        """

        logits = self.linear(mention_embeddings)
        # activations = logits
        if span_classes is not None:
            targets = span_classes
            # loss_function = nn.BCELoss()
            # torch.nn.BCELoss is unsafe to autocast
            loss_function = nn.BCEWithLogitsLoss()
            loss = loss_function(logits, targets)

            # 🔭🌕 Galileo logging
            # Only log if IDs are present
            if entity_ids is not None:
                task_mask = self.preprocessor.lookups.label_subset_arr

                task_specific_logits = logits.detach()
                task_specific_targets = targets.detach()
                # Check if we are logging all entity types
                if task_mask is not None:
                    task_specific_logits = task_specific_logits[:, task_mask]  # Shape = [batch, task_mask_shape]
                    task_specific_targets = task_specific_targets[:, task_mask]
                else:  # Remove the dummy 0 entity type when logging all types
                    task_specific_logits = task_specific_logits[:, 1:]
                    task_specific_targets = task_specific_targets[:, 1:]

                # Get the spans that actually have labels!
                # Note: when we're logging all types, likely all entities will have 1+ entity type.
                active_spans_mask = torch.where(torch.sum(task_specific_targets, dim=-1) > 0)[0]
                if active_spans_mask.shape[0] != 0:
                    task_specific_logits = task_specific_logits[active_spans_mask]
                    mention_embeddings = mention_embeddings[active_spans_mask]
                    entity_ids = entity_ids[active_spans_mask]

                    dq.log_model_outputs(
                        embs=mention_embeddings,
                        logits=task_specific_logits,
                        ids=entity_ids
                    )

            return loss, logits.sigmoid()

        return None, logits.sigmoid()

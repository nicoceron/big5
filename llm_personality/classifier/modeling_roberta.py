import torch
from torch import nn
from transformers import RobertaModel, RobertaPreTrainedModel
from typing import List, Optional, Tuple, Union, Dict, Any
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
import logging

logger = logging.getLogger(__name__)

class RobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier_O = RobertaClassificationHead(config)
        self.classifier_C = RobertaClassificationHead(config)
        self.classifier_E = RobertaClassificationHead(config)
        self.classifier_A = RobertaClassificationHead(config)
        self.classifier_N = RobertaClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels_openness: Optional[torch.FloatTensor] = None,
        labels_conscientiousness: Optional[torch.FloatTensor] = None,
        labels_extraversion: Optional[torch.FloatTensor] = None,
        labels_agreeableness: Optional[torch.FloatTensor] = None,
        labels_neuroticism: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels_* (`torch.FloatTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the regression loss for each personality trait.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = [
            self.classifier_O(sequence_output),
            self.classifier_C(sequence_output),
            self.classifier_E(sequence_output),
            self.classifier_A(sequence_output),
            self.classifier_N(sequence_output)
        ] # [tensor, tensor, tensor, tensor, tensor]
        # tensor shape [..., num_labels]
        
        # Collect all labels
        labels_list = [
            labels_openness,
            labels_conscientiousness, 
            labels_extraversion,
            labels_agreeableness,
            labels_neuroticism
        ]
        
        # Check if we have any labels for loss calculation
        have_labels = any(label is not None for label in labels_list)
        
        loss = None
        if have_labels:
            loss_values = []
            
            for i, (logit, label) in enumerate(zip(logits, labels_list)):
                if label is not None:
                    # Move labels to correct device
                    label = label.to(logit.device)
                    
                    # Use MSE loss for regression
                    loss_fct = MSELoss()
                    
                    # Ensure shapes match
                    if self.num_labels == 1:
                        loss_value = loss_fct(logit.squeeze(), label.squeeze())
                    else:
                        raise ValueError("Only single label regression (num_labels=1) is supported.")
                    
                    loss_values.append(loss_value)
                    logger.debug(f"Loss for trait {i}: {loss_value.item()}")
            
            # Average over all available losses
            if loss_values:
                loss = sum(loss_values) / len(loss_values)
                logger.debug(f"Average loss: {loss.item()}")
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
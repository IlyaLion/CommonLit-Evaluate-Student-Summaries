from transformers import AutoConfig

def freeze(module):
    for parameter in module.parameters():
        parameter.requires_grad = False

def get_backbone_config(config):
    backbone_config = AutoConfig.from_pretrained(config.model.backbone_type, output_hidden_states=True)

    backbone_config.hidden_dropout = config.model.backbone.hidden_dropout
    backbone_config.hidden_dropout_prob = config.model.backbone.hidden_dropout_prob
    backbone_config.attention_dropout = config.model.backbone.attention_dropout
    backbone_config.attention_probs_dropout_prob = config.model.backbone.attention_probs_dropout_prob
    return backbone_config
import torch.nn as nn
from .pooling_layers import get_pooling_layer
from .utils import freeze, get_backbone_config
from transformers import AutoModel

class CommonLitModel(nn.Module):
    def __init__(self, config, backbone_config):
        super().__init__()
        self.config = config
        self.backbone_config = backbone_config
        self._build_model()

    def _build_model(self):
        self.backbone = AutoModel.from_pretrained(self.config.model.backbone_type, config=self.backbone_config)

        self.pool = get_pooling_layer(self.config, self.backbone_config)
        self.fc = nn.Linear(self.pool.output_dim, 2)

        self.init_weights(self.fc)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.backbone_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.backbone_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        feature = self.pool(inputs, outputs)
        output = self.fc(feature)
        return output
    
def get_model(config):
    backbone_config = get_backbone_config(config) 
    model = CommonLitModel(config, backbone_config=backbone_config)

    if config.model.gradient_checkpointing:
        if model.backbone.supports_gradient_checkpointing:
            model.backbone.gradient_checkpointing_enable()
        else:
            raise NotImplementedError(f'{config.model.backbone_type} does not support gradient checkpointing')

    if config.model.freeze_backbone:
        freeze(model.backbone)
    if config.model.freeze_embeddings:
        freeze(model.backbone.embeddings)
    if config.model.freeze_first_n_layers > 0:
        freeze(model.backbone.encoder.layer[:config.model.freeze_first_n_layers])
    if config.model.reinitialize_last_n_layers > 0:
        for module in model.backbone.encoder.layer[-config.model.reinitialize_last_n_layers:]:
            model.init_weights(module)

    return model
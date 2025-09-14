from loguru import logger
from transformers import (
    AutoConfig,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from llmc.utils.registry_factory import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY
class Whisper(BaseModel):
    def __init__(self, config, device_map=None, use_cache=False):
        super().__init__(config, device_map, use_cache)

    def build_model(self):
        self.model_config = AutoConfig.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.model_path)
        logger.info(f"self.model : {self.model}")

    def find_blocks(self, modality="audio"):
        # Whisper encoder is similar to a Transformer stack
        self.blocks = self.model.model.encoder.layers

    def find_embed_layers(self):
        self.embed_tokens = self.model.model.encoder._input_embed_layer

    def get_embed_layers(self):
        return [self.embed_tokens]

    def get_head_layers(self):
        # Decoder LM head
        return ["proj_out"] if hasattr(self.model, "proj_out") else ["lm_head"]

    def get_pre_head_layernorm_layers(self):
        return [self.model.model.encoder.layer_norm]

    def get_layers_except_blocks(self):
        return [
            self.embed_tokens,
            self.model.model.encoder.layer_norm,
            getattr(self.model, "proj_out", self.model.lm_head),
        ]

    def skip_layer_name(self):
        return ["lm_head", "proj_out"]

    def has_bias(self):
        return True

    def get_layernorms_in_block(self, block, modality="audio"):
        return {
            'self_attn_layer_norm': block.self_attn_layer_norm,
            'final_layer_norm': block.final_layer_norm,
        }

    def get_act_fn_in_block(self, block):
        return {"activation_fn": block.fc1.activation_fn}

    def get_attn_in_block(self, block):
        return {"self_attn": block.self_attn}

    def get_matmul_in_block(self, block):
        return {
            "self_attn.q_proj": block.self_attn.q_proj,
            "self_attn.k_proj": block.self_attn.k_proj,
            "self_attn.v_proj": block.self_attn.v_proj,
        }

    def get_softmax_in_block(self, block):
        return {"self_attn.softmax": block.self_attn}

    def __str__(self):
        return f"\nModel: \n{str(self.model)}"

    def batch_process(
        self, samples, calib_or_eval="eval", apply_chat_template=False, return_inputs=True
    ):
        """
        Turn raw audio samples into Whisper model-ready inputs.
        """
        # device = next(self.model.model.parameters()).device

        # for example in self.testdata:
        #     # Extract audio
        #     audio = example["audio"]["array"]
        #
        #     # Tokenize input
        #     inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)

        processor = self.processor  # WhisperProcessor

        inputs = []
        for audio_item in samples["audio"]:
            arr = audio_item["array"]
            sr = audio_item["sampling_rate"]

            input_features = processor(
                arr, sampling_rate=sr, return_tensors="pt"
            ).input_features

            inputs.append({"input_features": input_features})

        return inputs

    def get_subsets_in_block(self, block):

        subsets = []

        subsets.append({
            "layers": {
                "self_attn.q_proj": block.self_attn.q_proj,
                "self_attn.k_proj": block.self_attn.k_proj,
                "self_attn.v_proj": block.self_attn.v_proj,
            },
            "prev_op": [block.self_attn_layer_norm],
            "input": ["self_attn.q_proj"],
            "inspect": block.self_attn,
            "has_kwargs": True,
        })
        # If decoder, add cross-attention (detected by presence of encoder_attn)
        if hasattr(block, "encoder_attn"):
            subsets.append({
                "layers": {
                    "encoder_attn.q_proj": block.encoder_attn.q_proj,
                    "encoder_attn.k_proj": block.encoder_attn.k_proj,
                    "encoder_attn.v_proj": block.encoder_attn.v_proj,
                },
                "prev_op": [block.encoder_attn_layer_norm],
                "input": ["encoder_attn.q_proj"],
                "inspect": block.encoder_attn,
                "has_kwargs": True,
            })

        # Feedforward layer 1 (MLP)
        subsets.append({
            "layers": {"fc1": block.fc1},
            "prev_op": [block.final_layer_norm],
            "input": ["fc1"],
            "inspect": block.fc1,
            "has_kwargs": False,
            "is_mlp": True,
        })

        # Feedforward layer 2 (MLP)
        subsets.append({
            "layers": {"fc2": block.fc2},
            "prev_op": [block.fc1],
            "input": ["fc2"],
            "inspect": block.fc2,
            "has_kwargs": False,
            "is_mlp": True,
        })

        return subsets
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
        Expects a list of dicts with {"audio": np.ndarray, "sampling_rate": int}
        """
        assert calib_or_eval in ["calib", "eval"]
        assert not apply_chat_template

        audio_arrays = [s["audio"] for s in samples]
        sampling_rates = [s["sampling_rate"] for s in samples]
        assert len(set(sampling_rates)) == 1, "All audios must have same sampling rate"

        inputs = self.processor(
            audio_arrays, sampling_rate=sampling_rates[0], return_tensors="pt"
        )
        return inputs

    def get_subsets_in_block(self, block):
        return [
            {
                "layers": {
                    "self_attn.q_proj": block.self_attn.q_proj,
                    "self_attn.k_proj": block.self_attn.k_proj,
                    "self_attn.v_proj": block.self_attn.v_proj,
                },
                "prev_op": [block.layer_norm],
                "input": ["self_attn.q_proj"],
                "inspect": block.self_attn,
                "has_kwargs": True,
            },
            {
                "layers": {"fc1": block.fc1},
                "prev_op": [block.layer_norm],
                "input": ["fc1"],
                "inspect": block.fc1,
                "has_kwargs": False,
                "is_mlp": True,
            },
            {
                "layers": {"fc2": block.fc2},
                "prev_op": [block.fc1],
                "input": ["fc2"],
                "inspect": block.fc2,
                "has_kwargs": False,
                "is_mlp": True,
            },
        ]

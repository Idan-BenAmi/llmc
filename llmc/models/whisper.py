from loguru import logger
from torch import nn
from transformers import (
    AutoConfig,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from llmc.utils.registry_factory import MODEL_REGISTRY
from .base_model import BaseModel
import torch
from collections import defaultdict
from torch.nn import functional as F
import inspect

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
        self.blocks = self.model.model.encoder.layers + self.model.model.decoder.layers

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
        },
        )
        subsets.append({
            'layers': {'self_attn.out_proj': block.self_attn.out_proj},
            'prev_op': [block.self_attn.v_proj],
            'input': ['self_attn.out_proj'],
            'inspect': block.self_attn.out_proj,
            'has_kwargs': False,
        },
        )
        # If decoder, add cross-attention (detected by presence of encoder_attn)
        if hasattr(block, "encoder_attn"):
            subsets.append({
                "layers": {
                    "encoder_attn.q_proj": block.encoder_attn.q_proj,
                },
                "prev_op": ["encoder_outputs", block.encoder_attn_layer_norm],
                "input": ["encoder_attn.q_proj"],
                "inspect": block.encoder_attn,
                "has_kwargs": True,
            })
            subsets.append({
                "layers": {
                    "encoder_attn.k_proj": block.encoder_attn.k_proj,
                    "encoder_attn.v_proj": block.encoder_attn.v_proj,
                },
                "prev_op": ["encoder_outputs", block.encoder_attn_layer_norm],
                "input": ["encoder_attn.k_proj"],
                "inspect": block.encoder_attn,
                "has_kwargs": True,
            })
            subsets.append({
                'layers': {'encoder_attn.out_proj': block.encoder_attn.out_proj},
                'prev_op': [block.encoder_attn.v_proj],
                'input': ['encoder_attn.out_proj'],
                'inspect': block.encoder_attn.out_proj,
                'has_kwargs': False,
            },
            )

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

    def get_decoder_inputs(self):
        return self.decoder_input

    # Update catcher to not raise assert during forward, in order to enable iterative generation during calibration.
    def get_catcher(self, first_block_input):
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
                self.signature = inspect.signature(module.forward)

            def forward(self, *args, **kwargs):
                params = list(self.signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i > 0:
                        kwargs[params[i]] = arg
                first_block_input['data'].append(args[0])
                if 'output_router_logits' in kwargs:
                    assert kwargs['output_router_logits'] is False
                    kwargs.pop('output_router_logits')
                first_block_input['kwargs'].append(kwargs)
                # raise ValueError
                return self.module.forward(args[0], **kwargs)
        return Catcher

    # Update collect_first_block_input to include for generative multi modality model.
    @torch.no_grad()
    def collect_first_block_input(self, calib_data, padding_mask=None):
        first_block_input = defaultdict(list)
        decoder_input = defaultdict(list)
        max_tokens = 6

        Catcher = self.get_catcher(first_block_input)
        CatcherDecoder = self.get_catcher(decoder_input)

        if not self.use_cpu_to_save_cuda_mem_for_catcher:
            self.move_embed_to_device('cuda')
            if self.vision_model:
                self.vision_model.cuda()
            if self.vision_projector:
                self.vision_projector.cuda()
            if self.audio_model:
                self.audio_model.cuda()
            if self.audio_projector:
                self.audio_projector.cuda()
            self.blocks[0] = self.blocks[0].cuda()
        self.model.model.encoder.layers[0] = Catcher(self.model.model.encoder.layers[0])
        self.model.model.decoder.layers[0] = CatcherDecoder(self.model.model.decoder.layers[0])

        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

        for data in calib_data:
            data = {
                k: (v if torch.is_tensor(v) else v)
                for k, v in data.items()
            }
            try:
                if not self.mm_model:
                    # self.model(**data)
                    self.model.generate(**data, max_new_tokens=max_tokens-1, use_cache=False, do_sample=False)
                else:
                    self.mm_model.generate(**data, max_new_tokens=128, do_sample=False)
            except ValueError:
                pass
        self.first_block_input = {
            k: v[::2] for k, v in first_block_input.items()
        }
        # self.decoder_input = decoder_input
        self.decoder_input = {
            k: [v[i] for i in range(max_tokens - 1, len(v), max_tokens)]
            for k, v in decoder_input.items()
        }
        assert len(self.first_block_input) > 0, 'Catch input data failed.'
        if padding_mask:
            for idx in range(len(self.first_block_input['data'])):
                token_num = self.first_block_input['data'][idx].shape[1]
                if token_num != padding_mask[idx].shape[1]:
                    padding_mask[idx] = F.pad(
                        padding_mask[idx],
                        self.get_one_pad_setting(
                            self.tokenizer.padding_side,
                            token_num - padding_mask[idx].shape[1]
                        ),
                        value=1
                    )
        self.padding_mask = padding_mask
        if not self.use_cpu_to_save_cuda_mem_for_catcher:
            if self.vision_model:
                self.vision_model.cpu()
            if self.vision_projector:
                self.vision_projector.cpu()
            if self.audio_model:
                self.audio_model.cpu()
            if self.audio_projector:
                self.audio_projector.cpu()
            self.blocks[0] = self.blocks[0].cpu()
            self.move_embed_to_device('cpu')
        # self.blocks[0] = self.blocks[0].module
        self.model.model.encoder.layers[0] = self.model.model.encoder.layers[0].module
        self.model.model.decoder.layers[0] = self.model.model.decoder.layers[0].module
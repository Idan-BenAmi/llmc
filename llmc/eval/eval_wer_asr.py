import torch
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
import evaluate
from loguru import logger

from .eval_base import BaseEval


class ASREval(BaseEval):
    def __init__(self, model, config):
        # Call BaseEval init, but overwrite dataset handling
        super().__init__(model, config)
        assert self.eval_cfg["name"] == "librispeech_asr", \
            f"ASREval only supports LibriSpeech, got {self.eval_cfg['name']}"

        # # Load LibriSpeech test-clean (default subset)
        # logger.info("Loading LibriSpeech test-clean split...")
        # self.testdata = load_dataset("librispeech_asr", "clean", split="test")

        # Load WER metric
        self.metric = evaluate.load("wer")

    @torch.no_grad()
    def eval_func(self, model_llmc, dataset, seq_len, batch_size, eval_pos=None):
        logger.info("Starting ASR evaluation on LibriSpeech...")

        predictions, references = [], []
        processor = self.model.processor
        normalizer = EnglishTextNormalizer(english_spelling_mapping={})
        device = next(self.model.model.parameters()).device

        for example in self.testdata:
            # Extract audio
            audio = example["audio"]["array"]

            # Tokenize input
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)

            # Run model
            generated_ids = self.model.model.generate(inputs.input_features)

            # Decode
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            predictions.append(normalizer(transcription.lower()))
            references.append(normalizer(example["text"].lower()))

        # Compute WER
        wer = self.metric.compute(predictions=predictions, references=references)
        logger.info(f"WER on LibriSpeech: {wer:.4f}")
        return {"wer": wer}

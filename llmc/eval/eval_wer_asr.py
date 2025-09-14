import torch
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

        processor = self.model.processor
        device = next(self.model.model.parameters()).device

        def map_to_pred(batch):
            audio = batch["audio"]
            input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"],
                                       return_tensors="pt").to(device).input_features
            batch["reference"] = processor.tokenizer._normalize(batch['text'])

            with torch.no_grad():
                predicted_ids = self.model.model.generate(input_features)[0]
            transcription = processor.decode(predicted_ids)
            batch["prediction"] = processor.tokenizer._normalize(transcription)
            return batch

        result = self.testdata.map(map_to_pred)

        wer = self.metric.compute(references=result["reference"], predictions=result["prediction"])

        logger.info(f"WER on LibriSpeech: {wer:.4f}")
        return {"wer": wer}

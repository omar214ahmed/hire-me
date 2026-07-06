"""
ONNX Runtime (INT8) backed replacement for FlagEmbedding's FlagReranker.

Exposes the same call shape as FlagReranker.compute_score(pairs, normalize=True),
so pipeline/ranker.py doesn't need to change.
"""

from typing import List, Sequence, Union


class ONNXReranker:
    def __init__(self, model_dir: str, onnx_file: str = "model_quantized.onnx"):
        import torch
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        model_dir = str(model_dir)

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_dir,
            file_name=onnx_file,
            provider="CPUExecutionProvider",
            local_files_only=True,
        )

    def compute_score(
        self,
        pairs: Union[Sequence[str], Sequence[Sequence[str]]],
        normalize: bool = True,
        max_length: int = 1024,
        batch_size: int = 16,
    ):
        """Mirrors FlagReranker.compute_score(...): returns a float for a
        single pair, or a list[float] for multiple pairs."""
        if len(pairs) > 0 and isinstance(pairs[0], str):
            pairs = [pairs]

        scores = []
        for i in range(0, len(pairs), batch_size):
            batch = list(pairs[i : i + batch_size])
            queries = [p[0] for p in batch]
            passages = [p[1] for p in batch]

            inputs = self.tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            outputs = self.model(**inputs)
            logits = outputs.logits.view(-1)

            if normalize:
                logits = self._torch.sigmoid(logits)

            scores.extend(logits.detach().cpu().tolist())

        return scores[0] if len(scores) == 1 else scores

"""
ONNX Runtime (INT8) backed replacement for FlagEmbedding's BGEM3FlagModel.

Only implements what this codebase actually uses: dense embeddings via
CLS-token pooling + L2 normalization (this is what BGE-M3's official
implementation does for the "dense_vecs" output — sparse/ColBERT vectors
are not computed here since nothing downstream uses them).

Exposes the same call shape as BGEM3FlagModel.encode(), so
pipeline/models.py can swap the backend without touching jd_processor.py,
candidates.py router, or ranker.py.
"""

from typing import List, Union

import numpy as np


class ONNXBGEM3Embedder:
    def __init__(self, model_dir: str, onnx_file: str = "model_quantized.onnx"):
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        model_dir = str(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_dir,
            file_name=onnx_file,
            provider="CPUExecutionProvider",
            local_files_only=True,
        )

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 12,
        max_length: int = 512,
        **kwargs,
    ) -> dict:
        """Mirrors BGEM3FlagModel.encode(...)["dense_vecs"] shape."""
        if isinstance(sentences, str):
            sentences = [sentences]

        all_vecs = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            outputs = self.model(**inputs)

            # BGE-M3 dense embedding = CLS token, L2-normalized
            cls = outputs.last_hidden_state[:, 0]
            norm = cls / cls.norm(dim=1, keepdim=True)
            all_vecs.append(norm.detach().cpu().numpy())

        dense_vecs = np.concatenate(all_vecs, axis=0) if all_vecs else np.empty((0,))
        return {"dense_vecs": dense_vecs}

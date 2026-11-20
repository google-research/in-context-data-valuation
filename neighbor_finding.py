# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import hnswlib
import torch
import gc
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

def get_embeddings(
    texts: List[str], model_name: str = "Qwen/Qwen3-Embedding-4B", batch_size: int = 32
) -> np.ndarray:
    """Loads the model and generates embeddings for a list of texts."""
    print(f"Loading embedding model: {model_name}...")

    model = SentenceTransformer(
        model_name, trust_remote_code=True, model_kwargs={"torch_dtype": torch.bfloat16}
    )

    print(f"Generating embeddings for {len(texts)} texts across all available GPUs...")

    pool = model.start_multi_process_pool()
    embeddings = model.encode(texts, pool=pool, batch_size=batch_size)
    model.stop_multi_process_pool(pool)

    # Explicitly clear the model and release PyTorch's reserved VRAM so that it can be used by scoring model.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings

def build_hnsw_index(
    embeddings: np.ndarray,
    space: str = "cosine",
    ef_construction: int = 200,
    M: int = 16,
    ef: int = 50,
) -> hnswlib.Index:
    """Builds and populates an HNSWlib index with the provided embeddings."""
    num_elements, dim = embeddings.shape
    print(f"Building HNSW index for {num_elements} elements with dimension {dim}...")

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)
    index.add_items(embeddings, np.arange(num_elements))
    index.set_ef(ef)
    return index

def find_neighbors(
    index: hnswlib.Index, query_embeddings: np.ndarray, k: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the top-k nearest neighbors for the query embeddings."""
    print(f"Finding top {k} neighbors...")
    labels, distances = index.knn_query(query_embeddings, k=k)
    return labels, distances

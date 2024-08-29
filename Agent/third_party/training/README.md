Not yet included in pyproject.toml since we can't easily point to internal git repository. 

To apply patches:
```bash
bash patch.sh
```

# Patched vLLM to support Llama 2 huggingface weights as a backend.

To obtain the patch:

```bash
git clone our internal transformers repository
git checkout v0.2.1.post1_hf
git diff v0.2.1.post1 > vllm_huggingface_backend.patch
```

# Patched transformers to fix NANs in llama2.

To obtain the patch:

```bash
git clone our internal transformers repository
git checkout float16
git diff origin | sed -e 's|src/||g' > llama_float16.patch
```

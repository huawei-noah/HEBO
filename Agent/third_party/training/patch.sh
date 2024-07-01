site_package=$(python -c "import transformers; print('/'.join(transformers.__file__.split('/')[:-2]))")
patch -N -p1 --directory=${site_package} < llama_float16.patch

site_package=$(python -c "import vllm; print('/'.join(vllm.__file__.split('/')[:-2]))")
patch -N -p1 --directory=${site_package} < vllm_huggingface_backend.patch
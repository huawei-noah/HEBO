# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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
# ============================================================================


from setuptools import setup


REQUIRED_PKGS = [
	"torch==2.3.1",
	"transformers==4.42.3",
	"tokenizers==0.19.1",
	"datasets==2.20.0",
	"trl==0.11.4",
	"evaluate==0.4.2",
	"accelerate==0.31.0",
	"sentencepiece==0.2.0",
	"deepspeed==0.14.4",
	"bitsandbytes==0.43.1",
	"tqdm",
	"peft>=0.3.0",
	"tyro>=0.5.7",
	"omegaconf",
	"tensorboard",
	"sacrebleu==2.4.2",
	"sqlitedict",
	"scikit-learn==1.5.1",
	"pytablewriter",
	"pycountry",
	"langdetect",
	"more-itertools==10.3.0",
	"immutabledict==4.2.0",
	"rouge_score==0.1.2",
	"nltk==3.8.1",
]


setup(
	name="SparsePO",
	version="0.0.1",
	python_requires=">=3.8",
	author="Fenia Christopoulou, Ronald Cardenas Acosta",
	readme="README.md",
	packages=["src"],
	install_requires=REQUIRED_PKGS
)
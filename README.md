<p align="center">
  <strong>CRISPR-Cas9 off-target prediction engine</strong>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b271f8cb-46e9-4266-9cdf-834f5cabf109" width="400"/>
</p>

## Problem Statement

### General Task

The CRISPR-Cas system is an RNA-guided nuclease commonly used to introduce a
single-stranded break (ssDNA nick) or double-stranded break (dsDNA break). The
guide RNA is typically 20 nucleotides (nts) long, and the genomic locus must
often be adjacent to a PAM (Protospacer Adjacent Motif) site for DNA cleavage to
occur.

The RNA sequence and PAM site determine the specificity of the nuclease.
However, many Cas and Cas-like nucleases exhibit off-target activity, meaning
they can occasionally bind to unintended sites resembling the target sequence.
When the CRISPR-Cas system localizes to such unintended sites and performs its
function, this is referred to as an off-target effect.

In in vitro methods, if off-target editing occurs in protein-coding regions of a
gene, it can lead to significant unintended consequences. Thus, predicting the
likelihood of off-target effects is critical.

This problem cannot be solved solely by algorithmic sequence alignment methods,
as off-target effects depend not only on nucleotide complementarity between the
RNA and DNA but also on more complex mechanisms.

Machine learning (ML) approaches are effective because, when trained on large
datasets of experimentally observed off-target effects, models learn to capture
hidden factors influencing targeting.

### Technical Task and data format

**Input**: Two sequences, sgRNA and targetDNA, represented as strings of
nucleotides (A, T, G, C) approximately 20 characters in length (variable size).

**Output**: A binary prediction (0 or 1) indicating the presence (1) or absence
(0) of an off-target effect.

### Metrics

Models in this project are evaluated via **Accuracy**, **AUROC**, **Precision**
and **Recall**.

the main CRISPR-BERT model is expected to perform around following:

| Model                                  | Val Accuracy | Val AUROC | Val Precision | Val Recall |
| -------------------------------------- | ------------ | --------- | ------------- | ---------- |
| Bidirectional LSTM with BERT embedding | 0.86         | 0.90      | 0.77          | 0.56       |

### Models

The main model is CRISPR-BERT based on architecture suggested in the paper
**Orhan Sari, Ziying Liu, Youlian Pan, Xiaojian Shao, Predicting CRISPR-Cas9
off-target effects in human primary cells using bidirectional LSTM with BERT
embedding, Bioinformatics Advances, Volume 5, Issue 1, 2025, vbae184,
https://doi.org/10.1093/bioadv/vbae184
(https://academic.oup.com/bioinformaticsadvances/article/5/1/vbae184/7934878)**

Stack encoding was used to encode the CRISPR-Cas9 sgRNA–DNA sequence pairs,
after which the data is passed into a BERT embedding layer. The output of BERT
is fed into a BiLSTM layer. The output of the BiLSTM is then passed through
dense layers with a sigmoid activation at the end to predict the presence of
off-target effects.

### Implementation

Final project runs as an ML Flow inference server with interactive frontend.
Please follow instructions below to run it locally on your machine.

## Setup

To run the program, follow these steps:

0. **Clone repository and enter project dir:**

   ```sh
   git clone https://github.com/tenebrissilvam/CRISPR-off-t-engine.git
   cd /CRISPR-off-t-engine
   ```

1. **Activate the Virtual Environment:** First, you need to activate the virtual
   environment of `uv`. This ensures that all the necessary dependencies are
   available.

   ```sh
   pip install uv

   uv venv --python $(cat .python-version) --prompt crispr-off-t-engine

   source .venv/uv/bin/activate

   uv sync
   ```

## Data

0. You can download csv file of Change_seq dataset from the following link:
   https://1drv.ms/x/c/695ce78a94063f8c/EbdYXZVnrT5ElMKf6cKtlugBznQbWkVrqr7tNWPrDLpcmg?e=iHYCsy

and paste it in the project directory ./data/Change_seq.csv

1. Or you can download it automatically via script to data/Change_seq.py

   ```sh
   uv run download_data.py
   ```

## Training

2. **Training** After environment preparation navigate to conf folder, switch
   label field at conf/mode/default.yaml to train and modify other parameters if
   needed, then run following command from the root of the repository to train
   the model.

```sh
uv run src/run_scripts/train.py
```

## Production preparation

**Plots visualisation** To view loss plits and AUROC, recall and precision
metrics first navigate to conf/logging/default.yaml and specify your wandb
username and project name.Then run

```sh
uv run src/utils/data_visuals.py
```

**MlFlow server preparation** You need to build the model first via command

```sh
uv run mlflow-serve/mlflow/mlflow_model_wrapper.py
```

**ONNX model conversion** To convert model into onnx format using pretrained
weights run following command to save model to
inference_model_modifications/onnx/crispr_detector.onnx

```sh
uv run inference_model_modifications/onnx/onnx_model_conversion.py
```

**TensorRT** 0. First pull docker image via

```sh
docker pull nvcr.io/nvidia/tensorrt:23.04-py3
```

1. Fix onnx model to have int32 weights via

```sh
cd /inference_model_modifications/onnx

uv run model_surgeon.py

cd ..

cd ..
```

2. then run interactive container

```sh
docker run -it --rm --gpus device=0   -v $(pwd)/inference_model_modifications/onnx:/models   nvcr.io/nvidia/tensorrt:23.04-py3
```

3. then inside the container run

```sh
trtexec \
  --onnx=/models/crispr_detector_fixed.onnx \
  --saveEngine=/models/crispr_detector_FP32.plan \
  --minShapes=input:1x22 \
  --optShapes=input:8x22 \
  --maxShapes=input:16x22 \
  --profilingVerbosity=detailed \
  --builderOptimizationLevel=5 \
  --hardwareCompatibilityLevel=ampere+
```

## Infer

**Run the Script:** for model inference on the provided dataset put csv dataset
path in conf/data/default.yaml and switch label field at conf/mode/default.yaml
to train. Also make sure you placed model weights correct path at
conf/mode/default.yaml. Then run

```sh
uv run src/run_scripts/infer.py
```

**MlFlow inference**

3. Run mlflow server with the model via

   ```sh
   mlflow models serve -m mlflow-serve/mlflow/crispr_off_t_model -p 8888 --host 0.0.0.0 --no-conda
   ```

4. Run mlflow model in a docker via

   ```sh
    sudo docker build -t mlflow-app -f mlflow-serve/mlflow-serve/mlflow/Dockerfile .
    sudo docker run -d -p 8888:8888 --name mlflow-server mlflow-app
   ```

Then you can send requests using following format

```sh

     curl -X POST http://localhost:8888/invocations         -H "Content-Type: application/json"   -d '{
             "inputs": [
                 {"sequence":"GTCACCAATCCTGTCCCTAGTGG",
                 "Target sequence": "TAAAGCAATCCTGTCCCCAGAGT"
                 }
             ]
         }'

```

3. To run all services use

```sh
   sudo docker-compose -f mlflow-serve/docker-compose.yml up --build

```

Then navigate to the http://localhost:9000 in your browser to use user friendly
RNA and DNA site input to predict off-target effect

## Project progress

### June 2025

- **07.06.2025**
  - Added ONNX conversion
  - Added autodownload script for the data

### May 2025

- **10.05.2025**
  - Added frontend to ML Flow server

### April 2025

- **20.04.2025**
  - Added Ml Flow inference
  - Added Dockerised model
  - Added support for curl requests for the model

### March 2025

- **31.03.2025**

  - Added Hydra configs.
  - Created run script.
  - Integrated DVC for version control.

- **30.03.2025**

  - Added project description.
  - Implemented metrics for performance evaluation on the Change-seq dataset for
    all four methods.

- **23.03.2025**

  - Added R-Crispr method.
  - Added Deep-CNN method.

- **17.03.2025**

  - Added Crispr DipOff method code.

- **15.03.2025**
  - Added bidirectional LSTM with BERT embeddings method code.

## References

This code is based on methods presented in following papers:

1. [Lin J, Wong KC. Off-target predictions in CRISPR-Cas9 gene editing using deep learning. Bioinformatics. 2018 Sep 1;34(17):i656-i663. doi: 10.1093/bioinformatics/bty554. PMID: 30423072; PMCID: PMC6129261. ](https://pmc.ncbi.nlm.nih.gov/articles/PMC6129261/)
2. [Orhan Sari, Ziying Liu, Youlian Pan, Xiaojian Shao, Predicting CRISPR-Cas9 off-target effects in human primary cells using bidirectional LSTM with BERT embedding, Bioinformatics Advances, Volume 5, Issue 1, 2025, vbae184, https://doi.org/10.1093/bioadv/vbae184](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbae184/7934878)
3. [Md Toufi kuzzaman, Md Abul Hassan Samee, M Sohel Rahman, CRISPR-DIPOFF: an interpretable deep learning approach for CRISPR Cas-9 off-target prediction, Briefi ngs in Bioinformatics, Volume 25, Issue 2, March 2024, bbad530, https://doi.org/10.1093/bib/bbad530](https://academic.oup.com/bib/article/25/2/bbad530/7588687)
4. [Niu R, Peng J, Zhang Z, Shang X. R-CRISPR: A Deep Learning Network to Predict Off-Target Activities with Mismatch, Insertion and Deletion in CRISPR-Cas9 System. Genes (Basel). 2021 Nov 25;12(12):1878. doi: 10.3390/genes12121878. PMID: 34946828; PMCID: PMC8702036.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8702036/)

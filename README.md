# Fine-grained aircraft classification on FGVC-Aircraft

**Image Processing and Computer Vision (IPCV), University of Bologna. Assignment Module 2.**

We classify images into the **100 aircraft variants** of the
[FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) dataset (e.g. Boeing 737-200, 737-300, …, 737-900)
in two ways:

1. **Part 1:** a convolutional network designed and trained **from scratch**, validated with three ablation studies;
2. **Part 2:** an **ImageNet-pretrained ResNet-18**, fine-tuned first with the recipe of Part 1 (2A) and then with tuned
   hyperparameters (2B).

All the work lives in a single, self-contained notebook, [`assignment_module_two.ipynb`](assignment_module_two.ipynb).
This README summarizes what is in it, so the code does not have to be read to understand the results.

---

## Results at a glance

| Model | Parameters | Validation acc. | Test acc. |
|---|---:|---:|---:|
| Part 1: our network, from scratch (448×448) | 11.2M | 68.9% | **68.0%** |
| 2A: ResNet-18 pretrained, recipe of Part 1 (448×448) | 11.2M | 80.5% | **81.8%** |
| 2B: ResNet-18 pretrained, TrivialAugment, 100 epochs (448×448) | 11.2M | 81.1% | **82.8%** |

The targets set by the assignment were ≈50% (Part 1) and ≈70% (Part 2).

With the selected configurations **re-trained on train + validation**, which is the standard protocol of the dataset, the
test accuracy reaches **84.2%** from scratch and **90.5%** with fine-tuning (see [Train + validation](#train--validation)).

**Main takeaways**

- **Input resolution** is the most important choice in both parts. Going from 224×224 to 448×448 gives +6.2 test points
  from scratch and +9.4 with fine-tuning.
- **From scratch**, three components are essential, and all three ablation studies agree on them:
  - data augmentation (−23 to −31 points without it);
  - a long enough schedule (−19 to −29 with 40 instead of 100 epochs);
  - batch normalization (−14 to −20).
- **Also from scratch**, three components have little or no measurable effect in our setting: skip connections, label smoothing and
  dropout.
- **Pretraining** is worth ≈14 test points with the very same recipe.
- **The usual fine-tuning advice does not hold here.** Smaller learning rates for the pretrained layers hurt (≈−4 points),
  because aircraft variants are far from the ImageNet classes and the pretrained features must change substantially.
- **The amount of training data** matters more than any other choice: doubling it with train + validation gives
  +16 points from scratch.

---

## Running the notebook

The notebook is designed for **Google Colab**. Open it in Colab (*File → Open notebook → GitHub*) and choose
*Runtime → Run all*.

As published, the notebook **does not train anything** (`ALLOW_TRAINING = False`). Executed top to bottom, it:

1. downloads the dataset (≈2.75 GB) from a copy shared by link on Google Drive, falling back to the official Oxford server
   if that download fails;
2. downloads the **published results** of all 54 experiments: a zip shared by link that holds the configuration,
   training history and accuracies of every run, plus the checkpoints of the three final models;
3. rebuilds every table and plot from these files.

It needs no account and no access to our files, and takes a few minutes, mostly to download and preprocess the
dataset. A GPU is not required; it only speeds up a short sanity check.

### Configuration

All the switches are in the first code cell:

| Variable | Default | Meaning |
|---|---|---|
| `ALLOW_TRAINING` | `False` | `True` trains every experiment whose results are not available. Published runs are still loaded, never re-trained. |
| `DRIVE_BACKUP` | `False` | `True` saves runs and caches on your own Google Drive, so a run interrupted by a Colab disconnection resumes from its last checkpoint. Recommended when training. |
| `PUBLISHED_RUNS` | our zip | Google Drive file ID(s) of the published runs. Leave it empty to start from nothing. |
| `DATASET_GDOWN_ID` | our copy | Drive file ID of the dataset archive. Leave it empty to download from Oxford. |
| `USE_WANDB` | `False` | Optional live monitoring on Weights & Biases. Off in all our runs: plots are always built from the saved histories. |
| `AUTHOR` | `"simone"` | Prefix of the run folders you create. It only matters when training. |

Each section has its own `RUN_...` flag, so single groups of experiments can be skipped.

### Re-training

Set `ALLOW_TRAINING = True` and, ideally, `DRIVE_BACKUP = True`. To re-train an experiment that is already published,
pass `force=True` to `run_experiment` or set `PUBLISHED_RUNS = {}`.

Indicative costs per run: ≈8 min at 224×224, ≈18 min at 336×336 and ≈30 min at 448×448 for the network of Part 1 on an
NVIDIA L4 (≈25 min at 448×448 on an A100).

---

## Notebook structure

| Section | Content |
|---|---|
| **0. Setup** | Configuration, imports, reproducibility (`fix_random`) |
| **1. Dataset** | Download, in-memory cache, data exploration, preprocessing and augmentation |
| **2. Training utilities** | Optimizers and schedules, training loop, `run_experiment`, plots and tables, published runs |
| **Part 1** (P1.1–P1.11) | Network design, sanity checks, incremental experiments, three ablation studies, input resolution |
| **2A** | ResNet-18 fine-tuned with the recipe of Part 1 |
| **2B** | Length of training, learning rate, input resolution, TrivialAugment, final model |
| **Train + validation** | Final configurations re-trained on train + validation |
| **Conclusions** | Summary of the findings |
| **Appendix** | How the zip of published runs was created |

Every experiment is followed by an *Observations* cell that comments on its table and training curves.

---

## Dataset and data pipeline

**FGVC-Aircraft** (`torchvision.datasets.FGVCAircraft`, `annotation_level="variant"`) contains 10,000 images of 100
aircraft variants. The dataset is perfectly balanced: 3,334 / 3,333 / 3,333 images in train / validation / test, i.e. only
**≈33 training images per class**. Variants of the same family differ only in small details (fuselage length, engines,
windows), which makes the task *fine-grained*.

Preprocessing:

- **Copyright banner.** Every image has a 20-pixel copyright banner at the bottom, which is cropped away.
- **In-memory cache.** The original images are large JPEGs (800–1600 px wide), so decoding and resizing them at every epoch
  would make the CPU the bottleneck. Each split is resized once and kept in RAM:
  - as raw `uint8` tensors at a short side of 256 px (≈3 GB);
  - as JPEG (quality 95) for higher resolutions, ≈10× smaller, decoded on the fly by the data loader workers.

  Random augmentations are still applied at every epoch.
- **Normalization.** Images are normalized with the ImageNet statistics in every part, so the pretrained ResNet-18
  receives inputs from the distribution it was trained on.
- **Input geometry.** Images are ≈1.5× wider than tall, so three ways of obtaining the network input are considered:
  - `crop`: square center crop, which may cut nose and tail;
  - `squash`: resize to a square, which distorts the aircraft (only illustrated, not trained);
  - `rect`: resize to a 1:1.5 rectangle, which works because the network ends with global average pooling.
- **Augmentation levels.**
  - `basic`: `RandomResizedCrop` with an aspect ratio matched to the input geometry, plus horizontal flip;
  - `strong`: `basic` plus color jitter and random grayscale, motivated by airline liveries (a potential shortcut) and by
    the grayscale photos in the dataset;
  - `trivial`: `basic` plus TrivialAugmentWide.

Model selection always uses the **validation** split. The **test** split is evaluated once per run, on the
best-validation checkpoint, and only reported.

---

## Part 1: a network from scratch

### Architecture: `AircraftNet`

All Part 1 networks belong to one configurable family. Every ablation therefore changes a single key of the
configuration.

```
input (3 × 448 × 448)
 └─ stem:    3×3 conv, stride 2, 32 channels → BN → ReLU
 └─ stage 1: 2 blocks,  64 channels (first block has stride 2)
 └─ stage 2: 2 blocks, 128 channels (first block has stride 2)
 └─ stage 3: 2 blocks, 256 channels (first block has stride 2)
 └─ stage 4: 2 blocks, 512 channels (first block has stride 2)
 └─ head:    global average pooling → dropout 0.3 → linear (100 classes)
```

- **Block:** two 3×3 conv–BN–ReLU layers.
  - `"residual"` adds a skip connection, as in ResNet; when the shape changes the shortcut is a 1×1 convolution.
  - `"plain"` has no skip connection.
- **Initialization:** Kaiming (He) initialization for ReLU networks.
- **Size:** 11.2M parameters, about the same as ResNet-18 (11.2M with a 100-class head). This makes the comparison with
  Part 2 fair.
- **Input size:** global average pooling makes the network independent of the input size, so the same weights work at
  224, 336 or 448 px.

### Training recipe of the final model

| | |
|---|---|
| Input | 448×448 center crop, `strong` augmentation |
| Optimizer | SGD, Nesterov momentum 0.9, LR 0.05, weight decay 5·10⁻⁴, batch size 64 |
| Schedule | 100 epochs, cosine decay with 2 warm-up epochs (updated every step) |
| Regularization | label smoothing 0.1, dropout 0.3 |
| Other | mixed precision (AMP) |

### How we got there

We followed Karpathy's *A Recipe for Training Neural Networks*:

1. verify the pipeline;
2. increase capacity until the network can fit the training set;
3. regularize;
4. train longer;
5. tune the input.

Each row of the table below changes one thing with respect to the previous adopted model.

| Step | Change | Val. acc. | Test acc. |
|---|---|---:|---:|
| Sanity checks | Initial loss ≈ ln(100); one batch overfitted to ≈0 loss | | |
| P1.3 Baseline | Plain blocks, 1 per stage, 32–256 channels (1.2M params), 192×288 `rect` input, `basic` augmentation, 40 epochs | 34.5% | 34.5% |
| P1.4 Capacity | Residual blocks, 2 per stage (2.8M) | 39.2% | 37.8% |
| P1.5 Regularization | Twice as wide (11.2M) + strong augmentation + label smoothing + dropout | 39.9% | 40.0% |
| P1.6 Longer training | 100 epochs instead of 40 | 59.0% | 58.4% |
| P1.8 Geometry vs resolution | 224×224 center crop instead of 192×288 rectangle | 62.3% | 61.8% |
| P1.10 Resolution | 336×336 | 67.2% | 66.3% |
| P1.10 Resolution | **448×448 (final model)** | **68.9%** | **68.0%** |

Tried and not adopted:

- residual blocks with one block per stage (−8 points): every shortcut is a strided projection, never an identity;
- width without regularization;
- regularization on the smaller network, which underfits;
- LR 0.02, which converges too slowly in 40 epochs;
- MixUp/CutMix (−1.3), which would need a longer schedule;
- a 256×384 rectangle (61.6%): keeping nose and tail brings nothing once the resolution is matched.

The finding of P1.8 deserves a note. The center crop beat the rectangular input not because of its shape but because the
aircraft is seen at a **higher resolution**. This motivated the resolution study of P1.10, which we stopped at 448×448
because the gain had shrunk to +1.8 points, close to the run-to-run noise.

### Ablation study of the final model (P1.11)

Each run removes or changes **one** component of the final model. A second seed measures the noise: differences up to
≈1 point are not significant.

| Variant | Val. acc. | Δ val. | Test acc. |
|---|---:|---:|---:|
| **Final model** | **68.9%** | | **68.0%** |
| Different seed (noise) | 69.1% | +0.1 | 69.2% |
| 40 epochs instead of 100 | 39.7% | **−29.3** | 40.7% |
| No data augmentation | 45.6% | **−23.4** | 45.5% |
| No batch normalization | 49.0% | **−20.0** | 49.4% |
| 224×224 input | 62.3% | −6.6 | 61.8% |
| 336×336 input | 67.2% | −1.8 | 66.3% |
| No label smoothing | 67.9% | −1.1 | 68.3% |
| No dropout | 68.9% | 0.0 | 68.3% |
| Plain blocks (no skip connections) | 69.6% | +0.6 | 69.4% |

The same ablation was also run on two earlier reference models: the 192×288 rectangular model (P1.7) and the 224×224
model (P1.9). The conclusions are the same in all three studies:

- **Essential:** augmentation, schedule length and batch normalization.
- **Without batch normalization** even the training accuracy stays at 62%, so its role here is mainly to make
  optimization possible.
- **Little or no measurable effect:** skip connections, label smoothing and dropout. With strong augmentation and a long
  schedule, augmentation dominates the other regularizers. A network of this depth with batch normalization trains just
  as well without skip connections.

---

## Part 2: fine-tuning a pretrained ResNet-18

`torchvision.models.resnet18` with `ResNet18_Weights.IMAGENET1K_V1`. Only the final fully connected layer is replaced
(dropout followed by a linear layer with 100 outputs), and the whole network is fine-tuned.

### 2A: same hyperparameters as Part 1

The exact training recipe of the final Part 1 model, including its 448×448 input, gives **80.5% validation / 81.8% test**:
+11.6 / +13.8 points over the network trained from scratch. The pretrained network reaches 100% training accuracy around
epoch 50.

### 2B: tuning for fine-tuning

One hyperparameter at a time, selected on validation accuracy. After the first experiment the runs use 50 epochs to
halve their cost.

| Experiment | Variant | Val. acc. | Test acc. |
|---|---|---:|---:|
| Length of training | 100 epochs (2A) | 80.5% | 81.8% |
| | 50 epochs (new reference) | 80.2% | 80.8% |
| Learning rate (50 ep.) | LR 0.01 | 75.9% | 77.3% |
| | LR 0.1 | 78.6% | 79.1% |
| | LR 0.05 on the classifier / 0.005 on the backbone | 75.8% | 76.6% |
| Input resolution (50 ep.) | 224×224 | 70.8% | 71.0% |
| | 336×336 | 75.6% | 77.4% |
| | 512×512 | 79.9% | 81.8% |
| Augmentation | TrivialAugment, 50 epochs | 80.5% | 81.4% |
| | **TrivialAugment, 100 epochs (final 2B model)** | **81.1%** | **82.8%** |

Findings:

- **50 epochs are enough.** The second half of the schedule adds nothing significant for a pretrained network.
- **Discriminative fine-tuning and smaller learning rates hurt** (≈−4 points). This agrees with Kornblith et al. (2019),
  who found that on FGVC-Aircraft fine-tuning has a smaller edge over training from scratch than on most datasets, and
  with Li et al. (2020), who found that the best fine-tuning hyperparameters depend on how similar the source and target
  domains are.
- **Resolution matters most.** At the native 224×224 resolution of the pretrained network, the 2A recipe would lose
  ≈9 points (72.5% test with 100 epochs). Accuracy saturates at 448×448, and 512×512 brings nothing.
- **TrivialAugment** gives the best validation accuracy (+0.6 points over 2A, within the ≈1-point noise estimated in Part 1). It keeps training accuracy
  below 100% and profits from the longer schedule.

The recipe inherited from Part 1 was therefore already close to the best one we found. The two choices that matter most
for fine-tuning here, a high input resolution and a learning rate large enough to adapt the pretrained features, were
both already part of it.

---

## Train + validation

All choices were made on the validation split, so it could not be used for training. Once the choices were fixed, the
two final configurations were **re-trained on train + validation** (6,667 images, ≈67 per class), the standard protocol
of FGVC-Aircraft:

- the hyperparameters are exactly the ones selected on validation;
- the **last epoch** is kept, since there is no split left for selection;
- the test split is evaluated once.

| Model | Test acc., train only | Test acc., train + val | Δ |
|---|---:|---:|---:|
| Part 1, final model (448×448) | 68.0% | **84.2%** | +16.2 |
| 2B, final model | 82.8% | **90.5%** | +7.7 |

The network trained from scratch, which must learn all its features from the training images, benefits twice as much as
the pretrained one. These results use a different protocol (twice the training data), so they are reported separately
from the other experiments.

---

## Experiment infrastructure

Built for long trainings on Colab, where sessions disconnect and two people train in parallel:

- **`run_experiment(name, build_model, cfg)`** trains a configuration and saves it in its own folder
  (`config.json`, `history.json`, `results.json`, `best.pt`, `last.pt`).
  - If the run is already complete, locally or among the published runs, it only loads its results.
  - If the run was interrupted, it resumes from `last.pt`.
  - If a run with the same name but a **different** configuration exists, it stops with an error, so results are never
    silently mixed up.
- **Published runs.** `publish_runs()` collects all completed runs into one zip, and `fetch_published_runs()` downloads
  it with `gdown`. This is what lets anyone rebuild the whole notebook without training.
- **Plots and tables** (`plot_runs`, `results_table`, `ablation_report`) are always built from the saved histories, so
  every figure can be regenerated from the published runs.
- **Reproducibility.** Fixed seeds everywhere, plus an optional `DETERMINISTIC` mode for bit-exact cuDNN kernels.

**How we worked.** The notebook was developed by both team members and kept in sync through this GitHub repository. Each
member trained part of the experiments on their own Colab account, and the runs were then merged into the published zip.
This is why every run folder is prefixed by its author.

---

## Requirements

Python 3 with `torch`, `torchvision` (`transforms.v2`), `numpy`, `pandas`, `matplotlib`, `Pillow`, `tqdm` and `gdown`, all
preinstalled on Colab (`gdown` is installed automatically if missing). `wandb` is only needed if
`USE_WANDB = True`.

---

## References

- S. Maji, E. Rahtu, J. Kannala, M. Blaschko, A. Vedaldi. *Fine-Grained Visual Classification of Aircraft*. arXiv:1306.5151, 2013.
- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition*. CVPR 2016.
- K. He, X. Zhang, S. Ren, J. Sun. *Delving Deep into Rectifiers*. ICCV 2015.
- A. Karpathy. *A Recipe for Training Neural Networks*. Blog post, 2019.
- J. Howard, S. Ruder. *Universal Language Model Fine-tuning for Text Classification*. ACL 2018.
- H. Li, P. Chaudhari, H. Yang, M. Lam, A. Ravichandran, R. Bhotika, S. Soatto. *Rethinking the Hyperparameters for Fine-tuning*. ICLR 2020.
- S. Kornblith, J. Shlens, Q. V. Le. *Do Better ImageNet Models Transfer Better?* CVPR 2019.
- S. G. Müller, F. Hutter. *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation*. ICCV 2021.

---

## Authors

**Simone Magli** and **Francesco Giuseppe Ielo**, University of Bologna.

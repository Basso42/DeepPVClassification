# **Applied Statistics Project: Deep Learning for Individual Solar Installation Detection from Aerial Images**

Second-year project at ENSAE in partnership with [Réseau Transport d'Électricité (RTE)](https://en.wikipedia.org/wiki/RTE_(company)).

**Grade: 18/20**

---

## **Key Idea**
Estimating diffuse electrical production is a growing challenge for forecasting electricity production and consumption in French households. Many households install solar panels without being systematically recorded in administrative data[1]. This lack of exhaustive registration makes it difficult to accurately predict national electricity consumption and optimize electricity distribution ([RTE forecasts](https://www.rte-france.com/eco2mix/la-production-delectricite-par-filiere)).

To tackle this issue, we deploy two convolutional neural network models:
- **LeNet-5** (TensorFlow)
- **ResNet-18** (PyTorch)

These models classify aerial images extracted from Google Earth Engine, labeled by Kasmi et al. (2022)[1].

---

## **Repository Structure**

### **Jupyter Notebooks**
- `Stats_desc.ipynb` - Descriptive statistics of the Kasmi et al. dataset.
- `LeNET5.ipynb` - Implementation of the LeNet-5 CNN for detecting solar panels.
- `ResNet18-pre_trained.ipynb` - Utilizes a pre-trained ResNet-18, retrained on the BDPV database.
- `ResNet18-transfer_learning.ipynb` - Freezes ResNet-18's convolutional layers and retrains only the fully connected layers on the BDPV database.
- `ROC_PR_curve_ResNet18.ipynb` - Compares the performance of both ResNet-18 models using ROC and precision-recall curves.
- `delete_NA_img.ipynb` - Contains a function to remove Google images that include a "gray mask" (equivalent to missing data).

### **Folders**
- `src/` - Python modules for the project, including:
  - LeNet-5 architecture
  - Data loader
  - Confusion matrix generation
- `weights/` - Pre-trained model weights.
- `rapport.pdf` - Project report.

---

## **Execution Guidelines**
### **Running ResNet-18 Pre-trained Model**
The `ResNet18-pre_trained.ipynb` notebook should be executed on the [INSEE cloud](https://datalab.sspcloud.fr/home) using the `Jupyter-pytorch-gpu` service with:
- **Minimum Persistence:** 15GB
- **Dataset Download & Extraction:** [Kasmi et al. (2022)](https://www.nature.com/articles/s41597-023-01951-4)
- **GPU Configuration:** Adjust based on the machine setup.

For an example of a properly configured working session, see [this link](https://datalab.sspcloud.fr/launcher/ide/jupyter-pytorch-gpu?autoLaunch=true&resources.requests.memory=«37Gi»&resources.limits.memory=«115Gi»&resources.requests.cpu=«10400m»&persistence.size=«34Gi»&onyxia.friendlyName=«GPU_statapps»&git.repository=«https%3A%2F%2Fgithub.com%2FBasso42%2FDeepPVClassification.git»).

---

## **References**

1. [A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays and installation metadata](https://www.nature.com/articles/s41597-023-01951-4) - Kasmi et al., 2023
2. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) - He et al., 2015

<!-- Additional references (commented out) -->
<!-- [HyperionSolarNet: Solar Panel Detection from Aerial Images](https://arxiv.org/pdf/2201.02107.pdf) - Parhar et al., 2022 -->

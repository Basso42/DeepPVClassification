## **Applied Statistics Project: Deep Learning for Individual Solar Installation Detection from Aerial Images**

Second-year project at ENSAE in partnership with [Réseau Transport d'Electricité (RTE)](https://en.wikipedia.org/wiki/RTE_(company)). Grade: 18 out 20.
### **Key Idea**
Estimating diffuse electrical production is an issue of growing importance for the forecast of electrical production and consumption of French households. Indeed, more and more households are installing solar panels without it being exhaustively registered administrative data[1] making it more difficult to accurately predict national electricity consumption and thus optimize electricity distribution ([RTE forecasts](https://www.rte-france.com/eco2mix/la-production-delectricite-par-filiere)). We deploy here two convolutional neural network models (LeNet5 with TensorFlow and ResNet18[2] with PyTorch) to classify aerial images extracted from Google Earth Engine and labeled by Kasmi et al. (2022)[1].

### **Organization of the repository**

* The ```Stats_desc.ipynb``` file contains descriptive statistics of the Kasmi et al. dataset.
* In the ```LeNET5.ipynb``` file, we parameterize the LeNet5 CNN to best predict the presence of solar panels on images.
* In the ```ResNet18-pre_trained.ipynb``` file, we use a pre-trained ResNet18 architecture. The entire CNN is re-trained on the BDPV database.
* In the ```ResNet18-transfer learning.ipynb``` file, only the fully-connected layers of the ResNet18 are re-trained on the BDPV database, the others are frozen.
* In the ```ROC_PR_curve_ResNet18.ipynb``` file, we display the ROC and precision-recall curves of the two ResNet18 CNNs for comparison.
* The ```delete_NA_img.ipynb``` file contains a function for deleting Google images that are associated with a "gray mask": the equivalent of a missing value for numerical data.
* The ```src``` folder contains some of the python modules imported into the various notebooks: the LeNet5 architecture, the dataloader, the code for creating the confusion matrix, etc.
* The ```weights``` folder contains the weights of our models.
* ```rapport.pdf``` is the written project report.
 
**Nota bene**: The file ```ResNet18-pre_trained.ipynb``` should be executed on the [INSEE cloud](https://datalab.sspcloud.fr/home) with the ```Jupyter-pytorch-gpu``` service and more than 15GB of ```persistence``` in the advanced service settings in order to download and unzip the dataset from [Kasmi et al. (2022)](https://www.nature.com/articles/s41597-023-01951-4). You should also adjust the number of GPUs in the script based on the chosen machine configuration. Here is [an example of a working session](https://datalab.sspcloud.fr/launcher/ide/jupyter-pytorch-gpu?autoLaunch=false&resources.requests.memory=%C2%AB37Gi%C2%BB&resources.limits.memory=%C2%AB115Gi%C2%BB&resources.requests.cpu=%C2%AB10400m%C2%BB&persistence.size=%C2%AB34Gi%C2%BB&onyxia.friendlyName=%C2%AB1_GPU_Torch_pers%C2%BB&git.repository=%C2%ABhttps%3A%2F%2Fgithub.com%2FBasso42%2FDeepPVClassification.git%C2%BB](https://datalab.sspcloud.fr/launcher/ide/jupyter-pytorch-gpu?autoLaunch=true&resources.requests.memory=«37Gi»&resources.limits.memory=«115Gi»&resources.requests.cpu=«10400m»&persistence.size=«34Gi»&onyxia.friendlyName=«GPU_statapps»&git.repository=«https%3A%2F%2Fgithub.com%2FBasso42%2FDeepPVClassification.git»&init.personalInit=«https%3A%2F%2Fminio.lab.sspcloud.fr%2Fgamer35%2Fshells_scripts%2Fshell_script_statapp.sh»)https://datalab.sspcloud.fr/launcher/ide/jupyter-pytorch-gpu?autoLaunch=true&resources.requests.memory=«37Gi»&resources.limits.memory=«115Gi»&resources.requests.cpu=«10400m»&persistence.size=«34Gi»&onyxia.friendlyName=«GPU_statapps»&git.repository=«https%3A%2F%2Fgithub.com%2FBasso42%2FDeepPVClassification.git»&init.personalInit=«https%3A%2F%2Fminio.lab.sspcloud.fr%2Fgamer35%2Fshells_scripts%2Fshell_script_statapp.sh»).

### References ###
[1][A crowdsourced dataset of aerial images with annotated solar photovoltaic arrays and installation metadata](https://www.nature.com/articles/s41597-023-01951-4) Kasmi et al., 2023
[2][Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) He et al., 2015
[//]: <> ([HyperionSolarNet: Solar Panel Detection from Aerial Images](https://arxiv.org/pdf/2201.02107.pdf) Parhar et al., 2022)

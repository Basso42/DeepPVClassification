# statapps

## **Projet de statistiques appliquées : Deep learning pour la détection d'installations solaires individuelles à partir d'images aériennes**

Projet de deuxième année à l'ENSAE en partenariat avec [Réseau Transport d'Electricité (RTE)](https://fr.wikipedia.org/wiki/RTE_(entreprise))
### **Idée clé**
L'estimation de la production électrique diffuse est une problématique importante pour la prévision de la production et de la consommation électrique des ménages français réalisée par [RTE](https://www.rte-france.com/eco2mix/la-production-delectricite-par-filiere). 
Nous déployons ici deux modèles de réseaux de neurones convolutifs (LeNet5 et ResNet18) pour classifier des images aériennes extraites de Google Earth Engine et labellisées par [Kasmi et al. (2022)](https://www.nature.com/articles/s41597-023-01951-4).

### **Méthodologie**


**Nota bene** : Le fichier ```ResNet18_Cloud.ipynb``` doit être exécuté sur le [cloud de l'INSEE](https://datalab.sspcloud.fr/home) avec le service ```Jupyter-pytorch-gpu``` et plus de 15go de ```persistence``` dans les réglages avancés du service afin de pouvoir télécharger et dézipper le jeu de données de [Kasmi et al. (2022)](https://www.nature.com/articles/s41597-023-01951-4). Il faut également moduler le nombre de gpu dans le script en fonction de la configuration machine choisie. Voici [un exemple de service approprié](https://datalab.sspcloud.fr/launcher/ide/jupyter-pytorch-gpu?autoLaunch=false&resources.requests.memory=%C2%AB37Gi%C2%BB&resources.limits.memory=%C2%AB115Gi%C2%BB&resources.requests.cpu=%C2%AB10400m%C2%BB&persistence.size=%C2%AB34Gi%C2%BB&onyxia.friendlyName=%C2%AB1_GPU_Torch_pers%C2%BB&git.repository=%C2%ABhttps%3A%2F%2Fgithub.com%2FBasso42%2FDeepPVClassification.git%C2%BB).


### **References**


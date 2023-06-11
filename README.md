# statapps

## **Projet de statistiques appliquées : Deep learning pour la détection d'installations solaires individuelles à partir d'images aériennes**

Projet de deuxième année à l'ENSAE en partenariat avec [Réseau Transport d'Electricité (RTE)](https://fr.wikipedia.org/wiki/RTE_(entreprise))
### **Idée clé**
L'estimation de la production électrique diffuse est une problématique importante pour la prévision de la production et de la consommation électrique des ménages français réalisée par [RTE](https://www.rte-france.com/eco2mix/la-production-delectricite-par-filiere). 
Nous déployons ici deux 

### **Méthodologie**
On déploie en deux temps
Nota bene : Le fichier ```ResNet18_Cloud.ipynb``` doit être exécuté sur le [cloud de l'INSEE](https://datalab.sspcloud.fr/home) avec le service ```Jupyter-pytorch-gpu``` et plus de 15go de ```persistence``` dans les réglages avancés du service afin de pouvoir télécharger et dézipper le jeu de données de [Kasmi et al. (2022)](https://www.nature.com/articles/s41597-023-01951-4). Il faut également moduler le nombre de gpu dans le script en fonction de la configuration machine choisie.

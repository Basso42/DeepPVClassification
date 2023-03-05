import torch.nn as nn

#n_classes = dimension de l'espace de Y (2 comme on est en classification binaire)
class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        # Première partie du modèle, qui correspond aux couches convolutionnelles.
        #Remarque: padding: ajoute les 0 artificiels sur les bords du tenseur
        #padding = 1 400*400 -> 402*402
        #padding = 2 400*400 -> 404*404

        self.feature_extractor = nn.Sequential(
            ####Première couche de convolution####

            # 3 composantes par pixel (si noir et blanc: 1): in_channels = 3
            # 6 filtres: out_channels=6
            # Dimension du motif = 5
            # Stride (pas) = 1
            # Padding = 2
            # Activation tanh
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding = 2),
            nn.Tanh(),

            #Première couche de pooling
            nn.AvgPool2d(kernel_size=2, stride=2),

            #Résumé

            #Input: 400*400*3
            #Après padding: 404*404*3
            #Après convolution: 400*400*6 (400=404-5+1)
            #Après pooling: 200*200*6

        ####Deuxième couche de convolution####

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),

            #Deuxième couche de pooling

            nn.AvgPool2d(kernel_size=2, stride=2),

            #Résumé

            #Input: 200*200*6
            #Après convolution: 196*196*16 (196=200-5+1)
            #Après pooling: 98*98*16
        
        ####Troisième couche de convolution####

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh(),

            #Input: 98*98*16
            #Après convolution: 94*94*120 (94=98-5+1)

            
            nn.Flatten(start_dim = 1, end_dim = -1), # on applique un applatissement à la dernière couche pour renvoyer un vecteur de taille 94*94*120=1060320 au classifieur
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1060320, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        # probs = F.softmax(logits, dim=1)
        return logits# , probs
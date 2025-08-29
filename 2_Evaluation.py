# **************************************************************************
# INF7370 Apprentissage automatique 
# Travail pratique 3
# ===========================================================================

#===========================================================================
# Dans ce script, on évalue l'autoencodeur entrainé dans 1_Modele.py sur les données tests.
# On charge le modèle en mémoire puis on charge les images tests en mémoire
# 1) On évalue la qualité des images reconstruites par l'autoencodeur
# 2) On évalue avec une tache de classification la qualité de l'embedding
# 3) On visualise l'embedding en 2 dimensions avec un scatter plot


# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# La libraire responsable du chargement des données dans la mémoire
from keras.preprocessing.image import ImageDataGenerator

# Affichage des graphes et des images
import matplotlib.pyplot as plt

# La librairie numpy
import numpy as np

# Configuration du GPU
import tensorflow as tf

# Utlilisé pour charger le modèle
from keras.models import load_model
from keras import Model

# Utilisé pour normaliser l'embedding
from sklearn.preprocessing import StandardScaler

from keras import backend as K

# ==========================================
# ===============GPU SETUP==================
# ==========================================

# Configuration des GPUs et CPUs
config = tf.compat.v1.ConfigProto(device_count={'GPU': 2, 'CPU': 4})
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess);


# ==========================================
# ==================MODÈLE==================
# ==========================================

# Chargement du modéle (autoencodeur) sauvegardé dans la section 1 via 1_Modele.py
model_path = "Model.hdf5"
autoencoder = load_model(model_path)

# ==========================================
# ================VARIABLES=================
# ==========================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                       QUESTIONS
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 1) A ajuster les variables suivantes selon votre problème:
# - mainDataPath
# - number_images
# - number_images_class_x
# - image_scale
# - images_color_mode
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# L'emplacement des images
mainDataPath = "vache_elephant/"

# On évalue le modèle sur les images tests
datapath = mainDataPath + "test"

# Le nombre des images de test à évaluer
number_images = 400 # 400 images
number_images_class_0 = 200 # 200 images pour la classe du chiffre 2
number_images_class_1 = 200 # 200 images pour la classe du chiffre 7

# Les étiquettes (classes) des images
labels = np.array([0] * number_images_class_0 +
                  [1] * number_images_class_1)

# La taille des images
image_scale = 128

# La couleur des images
images_color_mode = "rgb"  # grayscale ou rgb

# ==========================================
# =========CHARGEMENT DES IMAGES============
# ==========================================

# Chargement des images test
data_generator = ImageDataGenerator(rescale=1. / 255)

generator = data_generator.flow_from_directory(
    datapath, # Place des images d'entrainement
    color_mode=images_color_mode, # couleur des images
    target_size=(image_scale, image_scale),# taille des images
    batch_size= number_images, # nombre d'images total à charger en mémoire
    class_mode=None,
    shuffle=False) # pas besoin de bouleverser les images

x = generator.next()

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 2) Reconstruire les images tests en utilisant l'autoencodeur entrainé dans la première étape.
# Pour chacune des classes: Afficher une image originale ainsi que sa reconstruction.
# Afficher le titre de chaque classe au-dessus de l'image
# Note: Les images sont normalisées (entre 0 et 1), alors il faut les multiplier
# par 255 pour récupérer les couleurs des pixels
#
# ***********************************************

# Reconstruire les images avec l'autoencodeur
reconstructed_images = autoencoder.predict(x)

# Choisir une image de la classe 0 et une de la classe 1 pour l'affichage
index_class_0 = 0  # Index de la première image de la classe 0
index_class_1 = number_images_class_0  # Index de la première image de la classe 1 (après toutes les images de la classe 0)

# Afficher les images originales et reconstruites
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# Convertir les images de format [0, 1] en [0, 255] et changer le type à entier pour l'affichage
# Afficher l'image originale de la classe 0
axs[0, 0].imshow((x[index_class_0] * 255).astype('uint8'))
axs[0, 0].set_title('Original Class 0')
axs[0, 0].axis('off')

# Afficher l'image reconstruite de la classe 0
axs[0, 1].imshow((reconstructed_images[index_class_0] * 255).astype('uint8'))
axs[0, 1].set_title('Reconstructed Class 0')
axs[0, 1].axis('off')

# Afficher l'image originale de la classe 1
axs[1, 0].imshow((x[index_class_1] * 255).astype('uint8'))
axs[1, 0].set_title('Original Class 1')
axs[1, 0].axis('off')

# Afficher l'image reconstruite de la classe 1
axs[1, 1].imshow((reconstructed_images[index_class_1] * 255).astype('uint8'))
axs[1, 1].set_title('Reconstructed Class 1')
axs[1, 1].axis('off')

plt.show()

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 3) Définire un modèle "encoder" qui est formé de la partie encodeur de l'autoencodeur
# Appliquer ce modèle sur les images afin de récupérer l'embedding
# Note: Il est "nécessaire" d'appliquer la fonction (flatten) sur l'embedding
# afin de réduire la représentation de chaque image en un seul vecteur
#
# ***********************************************

# Défintion du modèle
input_layer_index = 0 # l'indice de la première couche de l'encodeur (input)
output_layer_index = 6 # l'indice de la dernière couche (la sortie) de l'encodeur (dépend de votre architecture)
# note: Pour identifier l'indice de la dernière couche de la partie encodeur, vous pouvez utiliser la fonction "model.summary()"
# chaque ligne dans le tableau affiché par "model.summary" est compté comme une couche
# encoder = Model(autoencoder.layers[input_layer_index].input, autoencoder.layers[output_layer_index].output)

encoder_input = autoencoder.layers[input_layer_index].input
encoder_output = autoencoder.layers[output_layer_index].output

# Vérifiez si la sortie est déjà un vecteur aplati
# Cela dépend de votre architecture d'autoencodeur spécifique
# Ajoutez une couche Flatten si nécessaire
from keras.layers import Flatten
encoder_output_flattened = Flatten()(encoder_output)

# Création du modèle d'encodeur
encoder = Model(inputs=encoder_input, outputs=encoder_output_flattened)

# Utilisation de l'encodeur pour obtenir les embeddings
embeddings = encoder.predict(x)

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 4) Normaliser le flattened embedding (les vecteurs recupérés dans question 3)
# en utilisant le StandardScaler
# ***********************************************

# Supposons que 'embeddings' contient les embeddings aplatis obtenus à partir de l'encodeur
scaler = StandardScaler()
embeddings_normalized = scaler.fit_transform(embeddings)

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 5) Appliquer un SVM Linéaire sur les images originales (avant l'encodage par le modèle)
# Entrainer le modèle avec le cross-validation
# Afficher la métrique suivante :
#    - Accuracy
# ***********************************************

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Flattening des images originales pour le SVM
x_flattened = x.reshape((x.shape[0], -1))

# Création du modèle SVM
svm_model = SVC(kernel='linear')

# Supposons que 'labels' contient les étiquettes de classe pour chaque image
# Application de la cross-validation
scores = cross_val_score(svm_model, x_flattened, labels, cv=5)

# Affichage de l'accuracy moyenne
print(f"Accuracy: {np.mean(scores):.2f}")

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 6) Appliquer un SVC Linéaire sur le flattened embedding normalisé
# Entrainer le modèle avec le cross-validation
# Afficher la métrique suivante :
#    - Accuracy
# ***********************************************

# Utilisation des embeddings normalisés pour l'entraînement du SVM
svm_model_normalized = SVC(kernel='linear')

# Application de la cross-validation sur les embeddings normalisés
scores_normalized = cross_val_score(svm_model_normalized, embeddings_normalized, labels, cv=5)

# Affichage de l'accuracy moyenne pour les embeddings normalisés
print(f"Normalized Embeddings Accuracy: {np.mean(scores_normalized):.2f}")

# ***********************************************
#                  QUESTIONS
# ***********************************************
#
# 7) Appliquer TSNE sur le flattened embedding afin de réduire sa dimensionnalité en 2 dimensions
# Puis afficher les 2D features dans un scatter plot en utilisant 2 couleurs(une couleur par classe)
# ***********************************************

from sklearn.manifold import TSNE

# Application de TSNE pour réduire à 2 dimensions
tsne = TSNE(n_components=2, random_state=42)
embeddings_tsne = tsne.fit_transform(embeddings_normalized)

# Affichage dans un scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], c=labels, cmap='viridis', alpha=0.5)
plt.title('TSNE des Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.colorbar(label='Classe')
plt.show()

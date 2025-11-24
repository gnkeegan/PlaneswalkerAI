# PlaneswalkerAI

### Análisis de cartas de Magic: The Gathering para formato standard

## Grant Nathaniel Keegan | A01700753

###### 

### Evidencia 2: Portafolio de Implementación

### Clase: Inteligencia artificial avanzada para la ciencia de datos II (Gpo 101)

### Instituto Tecnológico y de Estudios Superiores de Monterrey



Profesores:

Benjamín Valdés Aguirre | Carlos Alberto Dorantes Dosamantes

José Antonio Cantoral Ceballos | Eduardo Daniel Juárez Pineda

Ismael Solís Moreno

<p align="center">
  <img src="Banner.png" width="100%" alt="Portada del proyecto">
</p>

## Introducción


En mi evidencia para los módulos 2 y 3 de Inteligencia artificial avanzada para la ciencia de datos II, voy a implementar una red neuronal convolucional (CNN) para detectar con un alto grado de confianza todas las cartas en formato standard actuales del juego "Magic: The Gathering".



## Cómo correr el código - Preprocesamiento de Datos

Todos los códigos para descargar los datos y generar los metadatos, se deben de correr en una terminal. Estos incluyen: ***descargar_datos_csv.py*** para descargar el csv con los datos de todas las cartas standard. ***img_download.py*** para descargar las imágenes en la carpeta de elección, y ***metadata_generator.py*** para generar el csv con metadatos a partir del csv generado anteriormente.

El código tiene varias dependencias, de no tenerlas, debes introducir en la terminal de comandos:

pip install numpy

pip install pandas

pip install tqdm

pip install time

pip install tkinter



## Cómo correr el código - Entrenamiento


Los códigos de entrenamiento se ejecutan a través de 2 archivos .ipynb dentro de Google Colab. Para la ejecución efectiva del código, se necesitan descargar, y subir dentro de una carpeta en Google Drive (De preferencia de nombre RETO MAGIC):

El archivo "card_metadata.csv" el cual contiene los metadatos reales para el color de cada una de las cartas.

El dataset. La carpeta mtg_standard_cards contiene todas las cartas en formato standard (En ocubre de 2025). Que contiene 3,436 imágenes y pesa 296MB.

Los archivos: BaseModelCNN.ipynb y ModelResNet50.ipynb. Los cuales son el archivo con una CNN básica y el modelo mejorado usando ResNet50 respectivamente.

Una vez que estén los archivos.en la carpeta de Google Drive, Se ejecutan. Es importante que la ruta de los datos subida sea la correcta, ya que puede no ser la misma que en el notebook.


## Cómo correr el código - Clasificación

Para usar el modelo mejorado, se necesitan 2 archivos: ***clasificadorResNet50.py*** y el modelo ***MTGResNet50.keras*** en la misma carpeta. Una vez ejecutado el clasificador, se puede elegir la carpeta a clasificar. Como sugerencia, se debe de copiar la carpeta ***mtg_standard_cards*** para ejecutar el código sobre ella.

Dependencias - correr en consola en caso de no tenerlas:

pip install opencv-python

pip install numpy

pip install tensorflow

pip install pillow

pip install tkinter


## Documentación


Este proyecto incluye el archivo ***PlaneswalkerAI - Clasificador por color para Magic The Gathering - Grant Keegan*** donde documento el proceso de desarrollo y resultados de este proyecto a más detalle.



## Conclusión del proyecto



Este proyecto expandió mis conocimientos sobre aprendizaje de máquina, aprendizaje profundo y redes neuronales en más de una manera. Me ayudó a entender el funcionamiento de una red neuronal convolucional, y una aplicación efectiva para uso personal, y uno de mis pasatiempos favoritos.



### Notas y Referencias



A continuación, voy a anotar todos los recursos externos que utilicé para el desarrollo de mi proyecto, incluyendo ejemplos de código, teoría, y técnicas aprendidas en clase.



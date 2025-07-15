<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

# DiscoLazio Document classification 

## Indice

- [Introduzione](#introduzione)
- [Pipeline](#2-pipeline)
    - [Pulizia del dataset](#21-pulizia-del-dataset)
        - [Algoritmi](#211-algoritmi)
    - [Modello di riconoscimento](#22-modello-di-riconocimento)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 1. Introduzione

    L'applicazione deve usufuruire del dataset DiscoLazio per effettuare il riconoscimento dei documenti, pù precisamente vanno riconosciuti i documenti di tipo : 0, 1 e 2.
    Ogni indice rappresenta un tipo di documento preciso.

    La classificazione dei documenti in foto deve esser effettuata attraverso un modelli di intelligenza artificiale, dopo un primo sguardo al dataset i possibili modelli sono :
    - ResNet
    - DenseNet
    - ViT
    - PCE

    I modelli verranno testati tutti per vedere quale è il più affidibile

# 2. Pipeline

## 2.1 Pulizia del dataset

    Per prima cosa il dataset va pulito. Le best practice sono : 
    - Ridimensionamento delle immagini per avere una grandezza uniforme 
    - Eliminazione di Ombre 
    - Eliminazione di sfondi che non sono informativi riguardo al documento 
    - Riequilibrazione dei canali RGB 

    Per i documenti di tipo 2 verrà effettuata un operazione più elaborata, questo tipo di documento rappresenta ben 3 documenti in una sola foto, che possiamo rappresentare come una matrice alta e stretta. (A volte possono esserci più di 3 documenti)
    In questo caso spezzeremo l'immagine alta e stretta in sotto immagini che attraverseranno la pipeline di pulizia singolarmente per poi esser ricomposte in una sola immagine, le singole immagini non potranno avere la dimensione di altre immagini singole quindi verrà fatto un riequilibrio delle dimensioni.

### 2.1.1 Algoritmi 
    - Rilevamento bordi : Canny Edge Detection, Dilation, Probabilistic Hough
    - Rimozione ombre e riflessi : Morphological, Filtering, Illumination Correction
    - Rimozione sfondi non informativi : Contour Detection + Largest area approximation, GrabCut, DeepLbaV3 /U2-Net
    -Riequilibrio canali RGB : Histogram equilization, Color constancy

    Questa è una lista dei possibili algoritmi che si possono applicare, in fase di sviluppo si indentificherà il più adatto

## 2.2 Modello di riconocimento

    Una volta pulito il dataset si passerà al Fine Tuning di uno dei modelli prima citati.

    Una volta effettuato il fine tuning si procederà con test di inferenza per verificare l'efficacia del modello
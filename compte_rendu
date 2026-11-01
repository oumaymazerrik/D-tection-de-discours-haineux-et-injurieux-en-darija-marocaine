
# Compte rendu — Détection de discours haineux et injurieux en darija marocaine

**Groupe 1**  
**Étudiante :** Oumayma Zerrik  
**Dataset :** AfriHate — Moroccan Arabic (`ary`)  
**Environnement :** Kaggle Notebook — GPU NVIDIA Tesla T4

---

## 1. Introduction

Dans ce TP, j'ai étudié la détection automatique des discours haineux et injurieux en darija marocaine à travers plusieurs approches de Natural Language Processing (NLP).

L'objectif principal était de classifier les tweets en trois catégories :

- **Hate** : discours haineux.
- **Abuse** : langage injurieux ou offensant.
- **Normal** : contenu non haineux et non injurieux.

La difficulté principale réside dans la proximité entre Hate et Abuse, le déséquilibre des classes et les particularités linguistiques de la darija : écriture arabe, Arabizi, code-switching et absence de normalisation orthographique.

J'ai comparé des méthodes classiques de Machine Learning, des architectures Deep Learning et des modèles Transformers préentraînés.

## 2. Dataset et préparation

J'ai utilisé la configuration marocaine (`ary`) du dataset AfriHate, disponible sur Hugging Face.

### Répartition des données

| Split | Nombre de tweets |
|---|---:|
| Train | 3 240 |
| Validation | 695 |
| Test | 699 |
| **Total** | **4 634** |

### Distribution des classes — Train

| Classe | Nombre | Pourcentage |
|---|---:|---:|
| Abuse | 1 588 | 49,01 % |
| Normal | 1 182 | 36,48 % |
| Hate | 470 | 14,51 % |

L'ensemble d'entraînement ne contient aucune valeur manquante ni aucun doublon exact avant prétraitement.

J'ai appliqué un nettoyage léger comprenant le remplacement des URLs et mentions, la normalisation de certaines lettres arabes et la réduction des répétitions excessives.

J'ai conservé les textes originaux pour les Transformers afin de ne pas supprimer des informations linguistiques potentiellement utiles.

Après nettoyage, 2 600 tweets ont été modifiés et 62 doublons de textes nettoyés ont été observés.

### Analyse du vocabulaire

| Indicateur | Résultat |
|---|---:|
| Vocabulaire retenu | 6 554 |
| Mots uniques dans Train | 23 034 |
| OOV Train | 27,61 % |
| OOV Validation | 40,59 % |
| OOV Test | 39,47 % |
| Longueur des séquences | 46 mots |

Le taux OOV élevé sur Validation et Test indique une couverture lexicale limitée, particulièrement importante pour les modèles utilisant des embeddings de mots.

---

# 3. Réponses aux questions du compte rendu

## Q1. Distribution des labels et choix des métriques

J'ai observé un déséquilibre important dans les données d'entraînement. La classe Abuse représente 49,01 %, contre seulement 14,51 % pour Hate.

Sur Validation et Test, les distributions sont très proches :

| Classe | Validation | Test |
|---|---:|---:|
| Abuse | 48,92 % | 48,93 % |
| Hate | 14,68 % | 14,59 % |
| Normal | 36,40 % | 36,48 % |

Lors de l'analyse des systèmes d'écriture, j'ai également constaté que la distribution des classes varie entre les tweets Arabic, Latin et Mixed.

Dans une première analyse du Train, les tweets Latin comportaient 74,35 % de Abuse, contre 26,76 % pour les tweets Arabic. Cette analyse reste exploratoire, car les URLs et les mentions peuvent influencer la catégorisation des écritures.

J'ai également expérimenté une baseline Rule-Based basée sur un lexique d'insultes, qui a obtenu un Macro-F1 de 0,2650.

Ce résultat montre qu'une approche très simple ne suffit pas pour cette tâche.

**Métriques retenues :**

- Macro-F1 comme métrique principale.
- Hate F1 pour évaluer la classe minoritaire.
- PR-AUC par classe pour analyser le compromis Precision–Recall.

J'ai évité d'utiliser l'Accuracy comme critère principal, car un modèle peut obtenir une bonne précision globale tout en détectant mal les tweets Hate.

## Q2. Raccourcis (Partie A)

Dans la partie A, j'ai étudié les indices superficiels pouvant être corrélés aux labels.

Les principaux indices observés sont les insultes explicites, certaines expressions offensantes, les variations d'écriture et des indices lexicaux liés à certains sujets.

La baseline Rule-Based, utilisant un lexique d'insultes, a obtenu :

- Macro-F1 : 0,2650.
- Recall Abuse : 0,14.
- F1 Hate : 0,00.

Ce modèle reconnaît très peu de tweets injurieux et ne peut pas prédire Hate.

À l'inverse, les approches TF-IDF apprennent davantage de caractéristiques lexicales et obtiennent de meilleures performances.

Cependant, ces indices peuvent conduire à des erreurs. Par exemple, la présence d'un mot offensant dans une citation, une critique ou une discussion ne signifie pas nécessairement que le tweet constitue un discours haineux.

J'en conclus qu'un classifieur ne doit pas dépendre uniquement de mots isolés, mais également du contexte et de la cible du discours.

## Q3. Choix du prétraitement et ablations

J'ai privilégié un prétraitement léger pour conserver les caractéristiques de la darija.

### Prétraitement appliqué

- Remplacement des URLs.
- Remplacement des mentions.
- Suppression des diacritiques arabes.
- Normalisation légère des caractères.
- Réduction des répétitions excessives.
- Conservation de l'Arabizi et des mots français.

J'ai conservé les mots injurieux, car ils peuvent être utiles pour distinguer Abuse et Hate.

Pour mesurer l'importance de certains éléments textuels, j'ai réalisé une expérience d'ablation avec MARBERTv2 sur le Test.

| Transformation | Macro-F1 |
|---|---:|
| Texte original | 0,7668 |
| Sans emojis | 0,7703 |
| Sans ponctuation répétée | 0,7668 |
| Sans insultes sélectionnées | 0,7599 |
| Toutes les suppressions | 0,7624 |

La suppression des insultes sélectionnées entraîne une légère dégradation du Macro-F1.

En revanche, la suppression des emojis et de la ponctuation répétée a peu d'effet dans cette expérience.

Ces résultats suggèrent que les indices lexicaux jouent un rôle important, mais que MARBERTv2 utilise également d'autres informations.

Les 40,59 % d'OOV sur Validation montrent par ailleurs l'intérêt d'approches sous-lexicales, notamment FastText et les tokenizers de Transformers.

## Q4. Zero-Shot NLI (B1)

J'ai utilisé un modèle multilingual NLI préentraîné pour réaliser une classification Zero-Shot, sans fine-tuning sur AfriHate.

J'ai comparé trois formulations des hypothèses : arabe, français et anglais.

### Résultats sur Validation

| Langue | Macro-F1 | Hate F1 |
|---|---:|---:|
| Arabe | 0,4172 | 0,3434 |
| Français | **0,4675** | 0,3316 |
| Anglais | 0,4396 | 0,3052 |

Le français obtient le meilleur Macro-F1, tandis que l'arabe obtient le meilleur Hate F1.

La formulation des hypothèses influence donc les prédictions du modèle.

Cette sensibilité peut s'expliquer par les différences de formulations, de vocabulaire et de correspondance entre les hypothèses et les tweets.

Un modèle NLI peut classifier un texte sans entraînement spécifique à la tâche, car il estime la relation entre une prémisse et une hypothèse.

Toutefois, cela ne garantit pas une bonne compréhension des subtilités du dialecte marocain.

J'ai également constaté que les performances Zero-Shot restent inférieures à celles des modèles supervisés TF-IDF.

## Q5. DarijaToxicityDetector (B2)

J'ai évalué le modèle préentraîné `TypicaAI/DarijaToxicityDetector`, destiné à la détection de toxicité en darija.

Comme ce modèle effectue une classification binaire, j'ai utilisé la correspondance suivante :

- Abuse et Hate → offensive.
- Normal → clean.

### Résultats

| Métrique | Score |
|---|---:|
| Binary Macro-F1 | 0,5349 |
| AP Offensive | 0,7593 |
| Offensive Recall | 0,44 |
| Clean Recall | 0,70 |

Le modèle ne détecte que 195 des 442 tweets offensants.

Il manque donc 247 tweets offensants dans cette expérience.

Cette baisse de performance peut être liée au Domain Shift, puisque le modèle a été entraîné sur une plateforme et des données différentes de celles utilisées dans AfriHate.

Les différences de vocabulaire, de contexte et d'annotation peuvent également influencer les résultats.

Je ne compare pas directement son Macro-F1 binaire avec les scores des classifieurs à trois classes, car les tâches ne sont pas identiques.

## Q6. Gestion du déséquilibre des classes

J'ai étudié plusieurs moyens de prendre en compte le déséquilibre entre Abuse, Hate et Normal.

### Class Weights

J'ai testé des modèles TF-IDF avec et sans pondération des classes.

| Modèle | Macro-F1 | Hate F1 |
|---|---:|---:|
| Logistic Regression | 0,7171 | 0,4658 |
| Logistic Regression Balanced | 0,7938 | 0,6818 |
| Linear SVM | 0,7894 | 0,6286 |
| Linear SVM Balanced | 0,8001 | 0,6486 |

Ces résultats montrent que la pondération améliore les performances sur la classe Hate dans les configurations testées.

J'ai également utilisé des pertes pondérées dans plusieurs entraînements Deep Learning.

### Threshold Adjustment

Pour MARBERTv2, j'ai testé différents ajustements du score Hate sur Validation.

Le meilleur biais sélectionné était de +0,15, donnant :

- Macro-F1 Validation : 0,8350.
- Hate F1 Validation : 0,7474.

L'ajustement a ensuite été appliqué au Test sans recalibrage sur ce dernier.

### Sur-échantillonnage

Je n'ai pas réalisé de comparaison expérimentale complète avec le sur-échantillonnage.

Si cette technique est utilisée, elle doit être appliquée uniquement au Train, après la séparation des données.

En effet, effectuer le sur-échantillonnage avant la séparation peut introduire des exemples identiques ou très similaires dans Train et Validation, provoquant une fuite de données (Data Leakage).

## Q7. Comparaison des modèles C1 à C4

J'ai entraîné quatre architectures Deep Learning à partir du dataset marocain.

| Modèle | Macro-F1 | Hate F1 |
|---|---:|---:|
| BiLSTM | **0,7380** | **0,5936** |
| BiLSTM + Attention | 0,7152 | 0,5484 |
| Mini-Transformer | 0,6962 | 0,5182 |
| Character CNN | 0,7324 | 0,5852 |

Le BiLSTM obtient les meilleures performances globales parmi ces quatre configurations.

L'ajout de l'attention n'a pas amélioré les résultats du BiLSTM dans cette expérience.

Le Char-CNN obtient un Hate Recall de 0,77, ce qui montre son intérêt pour la détection de la classe minoritaire, même si sa précision sur Hate reste limitée.

### Attention et indices linguistiques

Dans la partie D2, j'ai examiné les poids d'attention sur cinq tweets.

Certains termes offensants ou liés à l'identité reçoivent des poids élevés.

Par exemple, dans un tweet correctement classifié Hate, le token « كافر » reçoit un poids d'attention d'environ 0,445.

Cependant, le modèle réalise également des erreurs lorsque certains mots sont interprétés sans suffisamment tenir compte du contexte.

Les poids d'attention indiquent sur quelles représentations le mécanisme se concentre, mais ne constituent pas une explication causale complète des décisions.

### Positional Encoding

J'ai également effectué une ablation du Positional Encoding dans le Mini-Transformer.

Dans l'expérience réalisée, le modèle sans Positional Encoding a obtenu de meilleurs résultats que celui avec Positional Encoding.

Ce résultat peut provenir de la taille limitée du dataset, du rôle des indices lexicaux ou de variations liées à l'entraînement.

Je considère cette observation comme exploratoire, car une comparaison répétée avec des initialisations et des seeds appariées serait nécessaire pour établir un effet fiable.

## Q8. PLM gelé vs Fine-Tuning

J'ai comparé deux manières d'utiliser les modèles préentraînés.

### PLM Frozen

Dans la partie D1, j'ai utilisé DarijaBERT gelé pour extraire des embeddings contextuels, puis entraîné une Logistic Regression.

| Méthode | Macro-F1 | Hate F1 |
|---|---:|---:|
| CLS + LR | 0,6850 | 0,4979 |
| Mean Pooling + LR | 0,6875 | 0,5145 |

Mean Pooling obtient des performances légèrement supérieures à CLS.

J'ai également utilisé les embeddings des tokens de DarijaBERT gelé avec un BiLSTM + Attention.

Cette combinaison atteint un Macro-F1 de 0,7848 et un Hate F1 de 0,6897.

### Full Fine-Tuning

Dans la partie E1, j'ai entraîné l'ensemble des paramètres des modèles Transformers.

| Modèle | Macro-F1 | Hate F1 | PR-AUC Hate |
|---|---:|---:|---:|
| DarijaBERT Fine-Tuning | 0,7973 | 0,6893 | 0,7150 |
| MARBERTv2 Fine-Tuning | **0,8265** | **0,7283** | **0,7818** |

Le Fine-Tuning permet une meilleure adaptation des représentations à la classification Hate/Abuse/Normal.

Dans les expériences réalisées, MARBERTv2 Fine-Tuning obtient le meilleur Macro-F1 sur Validation.

Cependant, cette approche nécessite plus de calculs et davantage de ressources mémoire qu'un PLM gelé.

## Q9. Multi-Task Learning et détection d'Arabizi

Dans la partie E2, j'ai implémenté une architecture Multi-Task avec deux têtes :

- Une tâche principale de classification Hate/Abuse/Normal.
- Une tâche auxiliaire de détection d'Arabizi.

La fonction de perte utilisée est :

`L_total = L_main + λ × L_aux`

J'ai comparé plusieurs valeurs de λ.

| Lambda | Macro-F1 | Hate F1 |
|---|---:|---:|
| 0,0 | **0,8036** | 0,7075 |
| 0,3 | 0,7978 | **0,7097** |
| 0,5 | 0,7951 | 0,7085 |
| 1,0 | 0,7780 | 0,6757 |

Le meilleur Macro-F1 est obtenu avec λ = 0, c'est-à-dire sans contribution de la tâche auxiliaire.

L'ajout de l'Arabizi ne produit donc pas d'amélioration globale dans les configurations testées.

Une tâche auxiliaire peut aider lorsqu'elle apporte des informations complémentaires utiles.

Elle peut également nuire à la tâche principale si ses annotations sont bruitées, si elle est faiblement liée à l'objectif principal ou si son poids est trop important.

Dans mon expérience, les labels auxiliaires ont été générés par des règles heuristiques, ce qui peut limiter leur qualité.

## Q10. Décalage de prior et réglage du seuil

J'ai comparé les distributions de Validation et Test :

| Classe | Validation | Test |
|---|---:|---:|
| Abuse | 48,92 % | 48,93 % |
| Hate | 14,68 % | 14,59 % |
| Normal | 36,40 % | 36,48 % |

Les proportions sont presque identiques.

Cela ne montre donc pas de décalage important des proportions de classes entre ces deux ensembles.

J'ai sélectionné le biais Hate de +0,15 sur Validation.

### Résultats

| Métrique | Validation | Test |
|---|---:|---:|
| Macro-F1 | 0,8350 | 0,7668 |
| Hate F1 | 0,7474 | 0,6354 |

Le Macro-F1 diminue de 0,0682 sur Test.

Cette baisse souligne qu'une optimisation réalisée sur Validation ne garantit pas les mêmes performances sur de nouvelles données.

Je n'ai pas recherché le seuil optimal a posteriori sur Test, car ce seuil ne devrait pas être utilisé pour sélectionner le modèle final.

### En production

Je mettrais en place :

- Un suivi de la distribution des classes prédites.
- Un monitoring du Macro-F1 et du Hate F1 sur des données annotées.
- Une surveillance du Data Drift.
- Un recalibrage des seuils sur de nouvelles données de validation représentatives.

Le Test doit rester indépendant de la sélection des hyperparamètres.

## Q11. Généralisation hors domaine et analyse par écriture

### Cross-Dialect Evaluation

Dans la partie F2, j'ai évalué MARBERTv2, entraîné sur la darija marocaine (`ary`), sur le dialecte algérien (`arq`).

| Configuration | Macro-F1 | Hate F1 |
|---|---:|---:|
| Avant adaptation | 0,6197 | 0,5083 |
| Après adaptation ARY + ARQ | **0,6632** | **0,5375** |

L'adaptation supplémentaire avec des données algériennes améliore les performances sur le Test algérien.

Cela montre l'importance du Domain Adaptation.

Cette évaluation est réalisée sans adaptation initiale au dialecte algérien, mais le modèle avait déjà été supervisé sur le dataset marocain : il ne s'agit donc pas d'un Zero-Shot NLI au sens de la partie B1.

### Analyse par système d'écriture

Sur le Test marocain, j'ai obtenu :

| Écriture | Macro-F1 | Hate F1 |
|---|---:|---:|
| Arabic | 0,7643 | 0,5846 |
| Mixed | 0,7630 | 0,6329 |
| Latin | 0,6606 | **0,7083** |

Le Macro-F1 est plus faible sur les tweets Latin.

Cependant, le Hate F1 est meilleur pour cette catégorie.

Cette différence suggère que les performances dépendent à la fois de l'écriture, des caractéristiques du contenu et de la distribution des classes.

### Évaluation avant déploiement

Pour évaluer un modèle sur des commentaires marocains réels, je construirais un échantillon annoté représentatif comportant différentes écritures, plateformes et catégories de discours.

J'évaluerais séparément le Macro-F1, le Hate F1, les PR-AUC et les performances par type d'écriture.

Je surveillerais également les changements linguistiques et les nouveaux usages de l'Arabizi.

## Q12. Analyse des erreurs et choix du modèle en production

### Analyse qualitative

Sur 699 tweets de Test, MARBERTv2 a commis 134 erreurs.

| Type de confusion | Nombre |
|---|---:|
| Normal → Abuse | 36 |
| Abuse → Normal | 28 |
| Hate → Normal | 21 |
| Hate → Abuse | 20 |
| Abuse → Hate | 17 |
| Normal → Hate | 12 |

J'ai également préparé une analyse qualitative préliminaire de 49 erreurs lisibles.

| Cause probable | Nombre |
|---|---:|
| Context Ambiguity | 16 |
| Class Overlap | 11 |
| Arabizi / Orthographic Variation | 9 |
| Possible Label Noise | 6 |
| Implicit Hate Speech | 4 |
| Code-Switching | 2 |
| Sarcasm / Irony | 1 |

Ces catégories sont des interprétations préliminaires et nécessitent une validation manuelle définitive.

Elles suggèrent que l'ambiguïté contextuelle, le chevauchement des classes et la variation orthographique figurent parmi les principales difficultés du modèle.

### Comparaison finale des modèles

J'ai comparé Linear SVM et MARBERTv2 sur le même ensemble Test.

| Métrique | Linear SVM | MARBERTv2 |
|---|---:|---:|
| Macro-F1 | 0,7620 | **0,7668** |
| Hate F1 | 0,6082 | **0,6354** |
| Latence | **0,268 ms/tweet** | 7,838 ms/tweet |
| Taille | **3,592 MB** | 621,222 MB |

MARBERTv2 obtient une légère amélioration de la qualité, particulièrement sur Hate.

Cependant, Linear SVM est environ 29 fois plus rapide dans les configurations matérielles testées et occupe beaucoup moins d'espace de stockage.

La latence du SVM a été mesurée sur CPU, tandis que celle de MARBERTv2 a été mesurée sur GPU T4. Il s'agit donc d'une comparaison indicative.

### Test de McNemar

J'ai effectué un test de McNemar entre les deux modèles.

La p-value obtenue est :

**p = 0,9196**

Comme cette valeur est supérieure à 0,05, je ne peux pas conclure à une différence statistiquement significative entre leurs proportions de prédictions correctes.

Le test de McNemar ne teste toutefois pas directement la différence de Macro-F1.

### Choix pour la production

Pour un environnement avec des ressources matérielles limitées et un besoin de faible latence, je choisirais **Linear SVM**.

Il offre des performances proches de MARBERTv2 avec un coût d'inférence et de stockage beaucoup plus faible.

Si la priorité est la qualité de détection de Hate et que les ressources sont suffisantes, je privilégierais MARBERTv2, sous réserve d'une validation supplémentaire sur des données représentatives de la production.

---

## 4. Reproductibilité et fiabilité

J'ai réalisé une expérience avec trois graines aléatoires sur FastText + BiLSTM.

| Seed | Macro-F1 | Hate F1 |
|---|---:|---:|
| 42 | 0,7740 | 0,6457 |
| 123 | 0,7633 | 0,6283 |
| 2026 | 0,7593 | 0,6180 |

Les résultats moyens sont :

- **Macro-F1 : 0,7655 ± 0,0076**
- **Hate F1 : 0,6307 ± 0,0140**
- **PR-AUC Hate : 0,6375 ± 0,0111**

Les écarts-types relativement faibles suggèrent une stabilité encourageante du modèle FastText + BiLSTM.

Cette expérience ne constitue cependant pas une validation multi-seed de MARBERTv2.

J'ai également calculé des intervalles de confiance Bootstrap sur le Test avec 1 000 rééchantillonnages.

| Métrique | Score | IC à 95 % |
|---|---:|---|
| Macro-F1 | 0,7668 | [0,7277 ; 0,8001] |
| Hate F1 | 0,6354 | [0,5497 ; 0,7059] |

L'intervalle du Hate F1 est plus large, ce qui indique une incertitude plus importante pour la classe minoritaire.

---

## 5. Conclusion générale

Ce TP m'a permis de comparer différentes approches de classification de discours haineux en darija marocaine.

J'ai constaté que les méthodes classiques TF-IDF + Machine Learning restent très compétitives, même face à des architectures Deep Learning et Transformers plus complexes.

Le Fine-Tuning de MARBERTv2 a obtenu le meilleur Macro-F1 sur Validation, avec **0,8265**, mais ses performances sur Test sont descendues à **0,7668** après ajustement de décision.

Linear SVM a obtenu un Macro-F1 Test de **0,7620**, avec un temps d'inférence et une taille de modèle beaucoup plus faibles.

Les expériences ont également mis en évidence plusieurs difficultés :

- La faible représentation de la classe Hate.
- La variation orthographique de la darija.
- Les confusions entre discours injurieux et haineux.
- Les différences entre dialectes arabes.
- La baisse de performance entre Validation et Test.

J'en conclus qu'un modèle NLP ne doit pas être évalué uniquement sur son score global. Il faut également considérer sa capacité à détecter les classes minoritaires, sa robustesse, son coût de calcul et sa capacité de généralisation.

Enfin, les analyses réalisées m'ont montré l'importance de la validation expérimentale, des tests statistiques et de l'analyse qualitative des erreurs pour mieux comprendre les limites d'un système de classification automatique.

---

## 6. Références

- [AfriHate Dataset — Hugging Face](https://huggingface.co/datasets/afrihate/afrihate)
- [DarijaBERT — SI2M-Lab](https://huggingface.co/SI2M-Lab/DarijaBERT)
- [MARBERTv2 — UBC-NLP](https://huggingface.co/UBC-NLP/MARBERTv2)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)

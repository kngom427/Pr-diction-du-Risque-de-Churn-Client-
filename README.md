# Prédiction du Risque de Churn Client

**Apprentissage supervisé — Classification binaire avec `tidymodels` et R**

Master 1 Informatique 

---

## Présentation du projet

Ce projet s'inscrit dans le cadre de l'apprentissage supervisé appliqué à un problème métier concret : la **détection anticipée du churn client** dans le secteur des télécommunications.

Le churn (ou attrition) désigne la résiliation du contrat d'un client. Son coût est considérable pour les entreprises : acquérir un nouveau client est cinq à dix fois plus coûteux que de fidéliser un client existant. Un modèle prédictif performant permet aux équipes commerciales et marketing d'agir de manière préventive, en ciblant les interventions de rétention sur les clients les plus à risque.

L'objectif technique est de construire, comparer et interpréter plusieurs modèles de classification capables de prédire si un client va résilier son contrat dans les 30 prochains jours.

---

## Données

| Propriété        | Valeur                                      |
|------------------|---------------------------------------------|
| Nom              | IBM Telco Customer Churn                    |
| Source           | [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Observations     | 7 043 clients                               |
| Variables        | 21 (démographiques, contractuelles, usage)  |
| Variable cible   | `churn` — binaire (Yes / No)                |
| Déséquilibre     | 73,5 % Non-churn / 26,5 % Churn             |

Les données sont publiques et ne contiennent aucune information personnelle réelle.

---

## Méthodologie

Le projet suit un pipeline structuré en six étapes :

**1. Analyse exploratoire (EDA)**
Distribution des variables, déséquilibre de classes, corrélations, visualisations comparatives selon le statut de churn.

**2. Prétraitement**
Nettoyage des valeurs manquantes, encodage one-hot des variables catégorielles, normalisation des variables numériques, gestion du déséquilibre par SMOTE (`themis`), le tout encapsulé dans une `recipe` reproductible de `tidymodels`.

**3. Modélisation**
Comparaison de quatre algorithmes via le framework `tidymodels` :

| Modèle                   | Package       | Hyperparamètres optimisés            |
|--------------------------|---------------|--------------------------------------|
| Régression logistique    | `glmnet`      | `penalty` (régularisation Lasso)     |
| Random Forest            | `ranger`      | `mtry`, `min_n`                      |
| XGBoost                  | `xgboost`     | `tree_depth`, `learn_rate`, `loss_reduction` |
| SVM à noyau radial       | `kernlab`     | `cost`, `rbf_sigma`                  |

**4. Validation croisée**
Validation croisée stratifiée à 5 plis (k-fold) avec recherche aléatoire des hyperparamètres (`tune_grid`).

**5. Evaluation**
Métriques retenues : AUC-ROC (métrique principale), F1-Score, Précision, Rappel, Accuracy.

**6. Interprétabilité**
- Importance globale des variables (impureté de Gini, Random Forest)
- Valeurs SHAP globales et par observation (`shapviz`)
- Explications locales LIME pour des cas individuels (`lime`)

---

## Résultats

Les performances obtenues sur le jeu de test (20 % des données, stratifié) sont les suivantes :

| Modèle                | AUC-ROC | F1-Score | Précision | Rappel |
|-----------------------|---------|----------|-----------|--------|
| XGBoost               | 0.8319   | 0.6083    | 0.5580     | 0.6684 |
| Random Forest         | 0.8263   | 0.5773    | 0.5526     | 0.6043  |
| Régression Logistique | 0.8268   | 0.5945    | 0.5178     | 0.7005  |
| SVM (RBF)             | 0.8179   | 0.5972    | 0.5027     | 0.7353  |

*Les valeurs exactes sont calculées et affichées dynamiquement dans le notebook lors de son exécution.*

**Principaux facteurs de risque identifiés :**
- Type de contrat mensuel (vs annuel ou biannuel)
- Ancienneté inférieure à 12 mois
- Charges mensuelles élevées relativement à l'ancienneté
- Absence de services de support technique (`tech_support = No`)
- Absence de sécurité en ligne (`online_security = No`)

---

## Structure du dépôt

```
churn-prediction-r/
│
├─ churn_prediction_colab.ipynb   # Notebook principal 
└── README.md ce fichier
```

---

## Installation et Reproduction

### Prérequis

- R >= 4.3.0
- RStudio >= 2023.09 (recommandé)

### Etapes

**1. Cloner le dépôt**

```bash
git clone https://github.com/<votre-username>/churn-prediction-r.git
cd churn-prediction-r
```

**2. Restaurer l'environnement de packages**

```r
# Dans la console R, depuis la racine du projet
install.packages("renv")
renv::restore()
```

Cette commande installe exactement les mêmes versions de packages que celles utilisées lors du développement.

**3. Télécharger les données**

Télécharger le fichier `WA_Fn-UseC_-Telco-Customer-Churn.csv` depuis [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

**4. Exécuter le notebook**

Exécuter le notebook en chargeant les données téléchargées sur google colab
---

## Compétences démontrées

Ce projet mobilise les compétences suivantes, directement applicables en contexte professionnel :

- Analyse exploratoire structurée et communication visuelle avec `ggplot2`
- Construction de pipelines de prétraitement reproductibles avec `tidymodels`
- Gestion du déséquilibre de classes (SMOTE)
- Optimisation d'hyperparamètres par validation croisée
- Comparaison rigoureuse de modèles sur un jeu de test indépendant
- Interprétabilité globale (SHAP) et locale (LIME)
- Reproductibilité de l'environnement avec `renv`
- Documentation et communication des résultats 

---



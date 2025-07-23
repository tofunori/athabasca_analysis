# Guide pas-à-pas : rendre votre notebook _multi-glaciers_ et automatiser l’analyse

Ce document décrit **comment transformer votre notebook `compare_albedo.ipynb`**
en un pipeline capable de traiter n’importe quel glacier à la demande, grâce à :

* un fichier `config.yaml` centralisant les métadonnées,
* la _cellule parameters_ + **Papermill** pour injecter les variables,
* un **Snakefile** minimal pour tout orchestrer (facultatif : un simple script bash si vous préférez).

---

## 0. Prérequis logiciels

| Outil | Rôle | Installation rapide |
|-------|------|--------------------|
| `conda`/`mamba` | gérer les environnements | `mamba install -c conda-forge snakemake papermill ruamel.yaml` |
| `papermill` | exécuter un notebook en passant des paramètres | *(installé ci-dessus)* |
| `snakemake` | orchestrer le workflow (optionnel mais conseillé) | *(installé ci-dessus)* |
| `yq` | lire/écrire du YAML en bash (si vous choisissez le script) | `mamba install -c conda-forge yq` |

---

## 1. Organiser le dépôt

```
project_root/
├── data/
│   ├── aws/               # séries CSV des stations
│   ├── masks/             # masques GeoTIFF
│   └── ...                # tuiles MODIS, etc.
├── notebooks/
│   └── compare_albedo.ipynb    # le notebook modèle (paramétrable)
├── executed/               # notebooks exécutés (git-ignorés)
├── results/                # CSV statistiques par glacier
├── figs/                   # figures par glacier
├── reports/                # HTML/PDF à partager
├── config.yaml             # métadonnées glaciers (voir §2)
└── Snakefile               # orchestration (voir §4)
```

---

## 2. Créer `config.yaml`

```yaml
glaciers:
  Athabasca:
    csv:   data/aws/Athabasca.csv
    mask:  data/masks/Athabasca.tif
    lat:   52.19
    lon:  -117.28

  Saskatchewan:
    csv:   data/aws/Saskatchewan.csv
    mask:  data/masks/Saskatchewan.tif
    lat:   52.12
    lon:  -117.03

  Bow:
    csv:   data/aws/Bow.csv
    mask:  data/masks/Bow.tif
    lat:   51.66
    lon:  -116.42
```

Ajoutez ou modifiez autant de glaciers que nécessaire ; ce fichier sera **l’unique
source de vérité** pour tous les chemins et coordonnées.

---

## 3. Paramétrer le notebook

1. Ouvrez `compare_albedo.ipynb`.
2. Insérez **une cellule en tout début** et donnez‑lui le tag `parameters`
   (menu : _View ▸ Cell Toolbar ▸ Tags_).

```python
# parameters
GLACIER   = "Athabasca"
CSV_PATH  = "data/aws/Athabasca.csv"
MASK_PATH = "data/masks/Athabasca.tif"
LAT       = 52.19
LON       = -117.28
```

3. Partout dans le notebook :

* remplacez les chemins « en dur » par `CSV_PATH`, `MASK_PATH`, etc.,
* préfixez vos sorties avec `GLACIER` :

```python
from pathlib import Path
OUT_DIR = Path("results") / GLACIER
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_stats.to_csv(OUT_DIR / "stats.csv", index=False)
plt.savefig(Path("figs") / GLACIER / "scatter.png")
```

---

## 4. Orchestration

### 4.1 Avec Snakemake (recommandé)

#### `Snakefile` minimal (15 lignes)

```python
configfile: "config.yaml"

SEL = config.get("glacier")               # option CLI
GLACIERS = [SEL] if SEL else list(config["glaciers"])

rule all:
    input: expand("reports/{g}.html", g=GLACIERS)

rule run_notebook:
    input: nb="notebooks/compare_albedo.ipynb"
    output:
        exec="executed/{glacier}.ipynb",
        html="reports/{glacier}.html"
    params: meta=lambda wc: config["glaciers"][wc.glacier]
    shell: """
        papermill {input.nb} {output.exec}             -p GLACIER {wildcards.glacier}             -p CSV_PATH  {params.meta[csv]}             -p MASK_PATH {params.meta[mask]}             -p LAT {params.meta[lat]}             -p LON {params.meta[lon]}
        jupyter nbconvert --to html {output.exec} --output {output.html}
    """
```

#### Exécuter

| Besoin | Commande |
|--------|----------|
| **Tous** les glaciers | `snakemake -j 8` |
| Un **seul** glacier | `snakemake -j 8 --config glacier=Saskatchewan` |
| Un fichier cible précis | `snakemake reports/Bow.html` |

### 4.2 Alternative bash (sans Snakemake)

`run_notebook.sh`

```bash
#!/usr/bin/env bash
set -e

GLACIER=$1
CSV=$(yq ".glaciers.$GLACIER.csv"   config.yaml)
MASK=$(yq ".glaciers.$GLACIER.mask" config.yaml)
LAT=$(yq ".glaciers.$GLACIER.lat"   config.yaml)
LON=$(yq ".glaciers.$GLACIER.lon"   config.yaml)

papermill notebooks/compare_albedo.ipynb executed/${GLACIER}.ipynb          -p GLACIER   "$GLACIER"          -p CSV_PATH  "$CSV"          -p MASK_PATH "$MASK"          -p LAT "$LAT" -p LON "$LON"

jupyter nbconvert --to html executed/${GLACIER}.ipynb                   --output reports/${GLACIER}.html
```

Usage : `./run_notebook.sh Saskatchewan`

---

## 5. Bonnes pratiques complémentaires

* **Environnements Conda par règle** : ajoutez `conda: envs/gdal.yml`
  dans vos règles Snakemake pour figer les versions.
* **Tests unitaires** : placez votre fonction `merge_terra_aqua()` dans
  `scripts/analysis.py` et couvrez‑la avec `pytest`.
* **Histogrammes/rapports** : si la génération de figures est lourde,
  séparez‑la dans une règle `make_plots` qui consomme `results/{g}/stats.csv`.
* **Cluster/HPC** : ajoutez `--profile slurm` à Snakemake, il parallélisera
  sans toucher au code.

---

## 6. Vérification rapide

1. `snakemake -n` (ou `./run_notebook.sh Bow --dry-run`)  
   → affiche le plan d’exécution, sans rien lancer.  
2. `snakemake -j 4 --config glacier=Bow`  
   → génère `reports/Bow.html`.  
3. Ouvrez le HTML : vérifiez que les stats et plots portent bien « Bow ».

---

### TL;DR

1. **`config.yaml`** : listez les CSV, masques, lat/lon par glacier.  
2. **Notebook** : une cellule `parameters` + variables pour remplacer
   les chemins codés en dur.  
3. **Papermill** exécute le notebook avec `-p` ; **Snakemake** (ou un
   script bash) boucle sur les glaciers et convertit en HTML.  
4. Lancez : `snakemake -j 8 --config glacier=Saskatchewan`  
   → les bonnes données sont chargées, et toutes les sorties sont
   automatiquement nommées pour Saskatchewan.

Bon déploiement !

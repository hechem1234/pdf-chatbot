# PDF Chatbot

Application locale de recherche documentaire : importez des PDF et des images, puis posez des questions sur leur contenu. Le backend utilise FastAPI, ChromaDB, Sentence Transformers et Ollama. L’interface est construite avec Angular.

## Fonctionnalités

- Importer et indexer des PDF contenant du texte extractible.
- Extraire le texte des images avec EasyOCR et créer une description visuelle avec Ollama Vision.
- Poser des questions en français sur les documents indexés et consulter les sources utilisées.

> Les PDF numérisés sans couche texte ne sont pas encore pris en charge par l’OCR PDF. Les images peuvent être analysées par OCR et par le modèle visuel.

## Prérequis

- Python 3.11 ou plus récent
- Node.js et npm
- Ollama installé et en cours d’exécution

Téléchargez les modèles Ollama utilisés par défaut :

```powershell
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b
```

Le premier démarrage peut aussi télécharger le modèle d’embeddings et les modèles OCR d’EasyOCR.

## Installation et démarrage

Ouvrez deux terminaux PowerShell depuis le dossier du projet.

### 1. Backend

Dans le premier terminal :

```powershell
cd .\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

L’API est disponible sur `http://localhost:8000`. Sa documentation interactive se trouve sur `http://localhost:8000/docs`.

### 2. Frontend

Dans le second terminal :

```powershell
cd .\frontend
npm install
npm start
```

Ouvrez `http://localhost:4200` dans votre navigateur. Le frontend communique avec l’API sur le port `8000`.

## Limites d’import

- PDF : 15 Mo maximum. Le PDF doit contenir du texte sélectionnable.
- Images : 10 Mo maximum. Formats acceptés : JPG, JPEG, PNG et WEBP.
- L’analyse visuelle des images nécessite le modèle Ollama `qwen2.5vl:3b`.

## Dépannage

- Si l’interface indique que l’API est hors ligne, vérifiez que le backend tourne sur le port `8000`.
- Si l’analyse d’image ne fonctionne pas, vérifiez qu’Ollama tourne et que les deux modèles requis sont installés.
- Si un PDF échoue sans texte extractible, il s’agit probablement d’un PDF numérisé. Convertissez-le avec OCR avant de l’importer.

## Dépôt

Les données locales (bases ChromaDB, images importées, caches Python, dépendances et fichiers compilés) sont exclues du dépôt par `.gitignore`.

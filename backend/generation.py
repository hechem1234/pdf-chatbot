import ollama
import re

from retrieval import retrieve
from config import LLM_MODEL
from image_service import targeted_analysis


def create_prompt(question, documents):

    context = ""

    for i, doc in enumerate(documents, start=1):

        page = f"Page : {doc['page']}" if doc.get("page") is not None else "Type : image"
        context += f"""
        [Source {i}]
        Fichier : {doc['source']}
        {page}
        Méthode : {doc.get('extraction_method', 'pdf_text')}

        Contenu :
        {doc['text']}

        ----------------------
        """

    prompt = f"""
    Tu es un assistant spécialisé dans les documents et images indexés.

    Instructions :
    - Réponds en français.
    - Utilise uniquement les informations du contexte.
    - N'invente aucune information.
    - Si le contexte ne permet pas de répondre,
      indique que l'information est absente.
    - Ajoute les références des sources utilisées
      sous la forme [Source 1], [Source 2], etc.

    CONTEXTE :
    {context}

    QUESTION :
    {question}

    RÉPONSE :
    """

    return prompt


def ask_rag(question):

    # 1. Recherche
    documents = retrieve(question)

    if not documents:
        return "Aucun document indexé ou passage trouvé.", []

    # Trigger a one-off image inspection only for questions asking for fine visual detail.
    visual_detail_question = re.search(
        r"\b(color|colour|couleur|position|left|right|top|bottom|gauche|droite|"
        r"exactly|precise|visible|shape|forme|where|which object|small detail)\b",
        question, re.IGNORECASE
    )
    if visual_detail_question:
        seen = set()
        for doc in list(documents):
            image_id = doc.get("document_id")
            if doc.get("type") != "image" or not image_id or image_id in seen:
                continue
            seen.add(image_id)
            try:
                detail = targeted_analysis(image_id, question)
                documents.append({
                    **doc,
                    "text": detail,
                    "extraction_method": "targeted_vision",
                    "page": None,
                })
            except Exception as error:
                print(f"Targeted vision analysis failed for {doc.get('source')}: {error}")

    # 2. Création du prompt
    prompt = create_prompt(question, documents)

    # 3. Génération
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.1
        }
    )

    answer = response.message.content

    # 4. Préparation des sources
    sources = []

    for doc in documents:

        source = {
            "file": doc["source"],
            "page": doc.get("page"),
            "type": doc.get("type", "pdf"),
            "document_id": doc.get("document_id"),
            "extraction_method": doc.get("extraction_method", "pdf_text"),
        }

        if source not in sources:
            sources.append(source)

    return answer, sources

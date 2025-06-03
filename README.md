# Street Fighter VI chatbot

Den här chatboten svarar på frågor om spelet Street Fighter VI. Informationen om spelet kommer från https://streetfighter.fandom.com/wiki/Street_Fighter_6

Detta är ett examinerande projekt i data science vid EC Utbildning AB.

## Modell och teknik

Chatboten använder Retrieval-Augmented Generation (RAG) för att hämta relevanta textstycken från en kunskapsbas och generera svar med hjälp av Googles språkmodell Gemini.  
Texten delas upp i mindre delar ("chunks") med olika metoder som utvärderas så att den bästa metoden med de bästa parametrarna kan användas i applikationen.

## Installation

1. Klona projektet:
   ```
   git clone https://github.com/SisselN/street_fighter_VI_chatbot.git
   ```
2. Installera beroenden:
   ```
   pip install -r requirements.txt
   ```

## Starta chatboten

Kör:
```
uvicorn backend.model_server:app --reload
```
```
python -m app.app
```

## Filbeskrivning

- **app/app.py** – Huvudfilen för Gradio-gränssnittet.
- **model_server.py** - Backend-servern (FastAPI) som tar emot frågor från frontend, hämtar relevanta textstycken med RAG och genererar svar med hjälp av Gemini-modellen.
- **chunking/** – Innehåller olika metoder för att dela upp texten i "chunks".
- **rag_pipeline.py** – Kod för RAG-modellen.
- **evaluate_chunking.py** – Script för att utvärdera de olika chunking-metoderna.
- **best_chunks.json** - Innehåller den optimala chunkingmetoden med de optimala parametrarna.
- **scraped_text.txt** – Textdata hämtad från Street Fighter-wikin.

## Hur hade chatboten kunnat användas i verkligheten?

En sådan typ av chatbot kan användas för att:
- Ge snabb och korrekt information om det den har kunskap om.
- Automatisera kundsupport eller FAQ.
- Hjälpa användare att hitta relevant information i stora textmängder.

## Utmaningar och säkerhet

- **Säkerhet:** Det är viktigt att filtrera och moderera användarfrågor och svar för att undvika olämpligt innehåll.
- **Datakvalitet:** Svaren är beroende av kvaliteten på den text som används som kunskapsbas.
- **Prestanda:** Långa texter och stora språkmodeller kan göra systemet långsamt.
- **Integritet:** Ingen personlig data lagras, men det är viktigt att informera användare om detta.

## Kontakt

Utvecklare: Sissel Nevestveit 
Mail: sisselnevestveit@hotmail.com

# Street Fighter VI chatbot

Den här chatboten svarar på frågor om spelet Street Fighter VI. Informationen om spelet kommer från https://streetfighter.fandom.com/wiki/Street_Fighter_6

Detta är ett examinerande projekt i data science vid EC Utbildning AB.

## Modell och teknik

Chatboten använder Retrieval-Augmented Generation (RAG) för att hämta relevanta textstycken från en kunskapsbas och generera svar med hjälp av en språkmodell.  
Texten delas upp i mindre delar ("chunks") med olika chunking-metoder, och den bästa metoden utvärderas med hjälp av `evaluate_chunking.py`.

## Installation

1. Klona projektet:
   ```
   git clone 
   ```
2. Installera beroenden:
   ```
   pip install -r requirements.txt
   ```

## Starta chatboten

Kör:
```
python app/app.py
```
och öppna webbläsaren på [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Filbeskrivning

- **app/app.py** – Huvudfilen för Gradio-gränssnittet.
- **chunking/** – Innehåller kod för att dela upp texten i chunks.
- **rag_pipeline.py** – Kod för RAG-modellen.
- **evaluate_chunking.py** – Script för att utvärdera olika chunking-metoder.
- **best_chunks.json** - Innehåller de optimala chunkingmetoden och-parametrarna.
- **scraped_text.txt** – Textdata hämtad från Street Fighter-wikin.

## Hur hade chatboten kunnat användas i verkligheten?

En sådan chatbot kan användas för att:
- Ge snabb och korrekt information om spelet.
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

import streamlit as st
import torch
import torch.nn.functional as F
import os
import re
from collections import Counter
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ddgs import DDGS
from groq import Groq

 # Wczytuje klucz API z pliku .env
load_dotenv()
klucz_groq = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Wykrywacz Dezinformacji", page_icon="logo.png", layout="wide")

# Ładowanie cache
@st.cache_resource # Żeby nie ładować modelu przy każdym kliknięciu
def zaladuj_model_lokalny():
    sciezka = "./moj_model_fake_news" 
    urzadzenie = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizator = AutoTokenizer.from_pretrained(sciezka)
    model = AutoModelForSequenceClassification.from_pretrained(sciezka)
    model.to(urzadzenie)
    return tokenizator, model, urzadzenie

tokenizator, model, urzadzenie = zaladuj_model_lokalny()

# Backend
def klasyfikuj_tekst(tekst):
    # Zamiana tekstu na liczby dla modelu
    wejscie = tokenizator(tekst, return_tensors="pt", truncation=True, max_length=512).to(urzadzenie)
    
    with torch.no_grad():
        wynik = model(**wejscie)
        procenty = F.softmax(wynik.logits, dim=1) # Zamiana na %
        
    # 0=PRAWDA, 1=FAŁSZ
    return procenty[0][0].item(), procenty[0][1].item()

import re

def szukaj_w_internecie(tekst):
    zapytanie = f"fact check {tekst[:150]}" 
    
    try:
        with DDGS() as wyszukiwarka:
            wyniki = list(wyszukiwarka.text(zapytanie, max_results=3))

            if wyniki:
                # Formatujemy wyniki w czytelny tekst
                return "\n".join([f"- {w['body']} (Link: {w['href']})" for w in wyniki])
            else:
                return "Brak wyników. DuckDuckGo nic nie znalazł dla tej frazy."
                
    except Exception as e:
        st.warning(f"Błąd połączenia z wyszukiwarką: {e}")
        return "Błąd modułu wyszukiwania."

def zapytaj_eksperta_ai(tekst, kontekst):
    if not klucz_groq:
        return "Brak klucza API w pliku .env"
    
    klient = Groq(api_key=klucz_groq)
    prompt = f"""
    Jesteś ekspertem weryfikacji informacji.
    Oceń wiarygodność newsa na podstawie znalezionego KONTEKSTU.
    Pisz krótko, zwięźle i po POLSKU.
    
    KONTEKST Z SIECI: {kontekst}
    NEWS DO OCENY: "{tekst[:800]}"
    """

    czat = klient.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
    return czat.choices[0].message.content

# Frontend
st.title("WYKRYWACZ DEZINFORMACJI")
st.markdown("Aplikacja łączy model **Deep learning (DistilBERT)** z weryfikacją LLM w **Internecie (RAG)**.")

# Dzielimy ekran na dwie kolumny
kolumna_lewa, kolumna_prawa = st.columns([1, 1])

with kolumna_lewa:
    tekst_uzytkownika = st.text_area("Wklej artykuł do sprawdzenia:", height=300)
    przycisk = st.button("Sprawdź wiarygodność tekstu", type="primary", use_container_width=True)

with kolumna_prawa:
    if przycisk and tekst_uzytkownika:
        if len(tekst_uzytkownika) < 10:
            st.warning("Wpisz dłuższy tekst.")
        else:
            # Model klasyfikujący
            with st.spinner("Analiza modelu statystycznego..."):
                p_prawda, p_falsz = klasyfikuj_tekst(tekst_uzytkownika)
            
            st.subheader("Wynik klasyfikacji")
            k1, k2 = st.columns(2)
            k1.metric("PRAWDA", f"{p_prawda:.1%}")
            k2.metric("FAŁSZ", f"{p_falsz:.1%}", delta_color="inverse")
            
            if p_falsz > 0.5:
                st.error("Model podejrzewa DEZINFORMACJĘ!")
            else:
                st.success("Tekst wygląda na WIARYGODNY.")
            
            st.divider()
            
            # RAG + LLM
            with st.spinner("Weryfikacja faktów w internecie..."):
                kontekst = szukaj_w_internecie(tekst_uzytkownika)
                opinia_eksperta = zapytaj_eksperta_ai(tekst_uzytkownika, kontekst)
            
            st.subheader("Analiza ekspercka AI")
            st.info(opinia_eksperta)
            
            with st.expander("Zobacz źródła weryfikacji"):
                st.write(kontekst)
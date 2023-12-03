from django.shortcuts import render
import requests
from bs4 import BeautifulSoup
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from mi_aplicacion.models import PaginaWeb, Parrafo, AnalisisTexto


def analisis_texto(request):
    
    url = "https://cnnespanol.cnn.com/2023/11/23/frente-frio-sureste-mexico-peninsula-yucatan-orix/"
    response = requests.get(url)
    html_content = response.text

    
    soup = BeautifulSoup(html_content, 'html.parser')

    
    paragraphs = soup.find_all('p')
    text = '\n'.join([paragraph.get_text() for paragraph in paragraphs])

   
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)

 
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  
    summary_text = ' '.join([str(sentence) for sentence in summary])

    return render(
        request,
        'mi_aplicacion/analisis_texto.html',
        {'tokens': tokens, 'text': text, 'tagged': tagged, 'summary': summary_text}
    )

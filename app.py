from flask import Flask, request, jsonify, render_template
import os
import logging
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import requests
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.callbacks.manager import get_openai_callback

# Sub-question engine için
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Wikipedia API
import wikipedia
import wikipediaapi

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask uygulaması
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

# Global değişkenler
vector_store = None  # Holds the FAISS vector database for semantic search
qa_chain = None      # Holds the RetrievalQA chain for question answering
sub_question_engine = None  # Holds the sub-question generation engine

load_dotenv()  # .env dosyasındaki değişkenleri yükle

class SubQuestion(BaseModel):
    question: str = Field(description="Alt soru")
    reasoning: str = Field(description="Bu sorunun neden sorulduğu")

class SubQuestionList(BaseModel):
    # Represents a list of sub-questions
    sub_questions: List[SubQuestion] = Field(description="Alt sorular listesi")

class TurkishCityRAG:
    def __init__(self):
        # Get the OpenAI API key from environment variables (.env file)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set!")
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        # Initialize LangChain components for embeddings, LLM, and text splitting
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.1)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        # Set up the Wikipedia API for Turkish language content
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='tr',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='TurkishCityRAG/1.0'
        )
        
        # List of all 81 Turkish cities (provinces) for the knowledge base
        self.turkish_cities = [
            "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Amasya", "Ankara", 
            "Antalya", "Artvin", "Aydın", "Balıkesir", "Bilecik", "Bingöl", 
            "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", "Çankırı", 
            "Çorum", "Denizli", "Diyarbakır", "Edirne", "Elazığ", "Erzincan", 
            "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", "Hakkari", 
            "Hatay", "Isparta", "Mersin", "İstanbul", "İzmir", "Kars", 
            "Kastamonu", "Kayseri", "Kırklareli", "Kırşehir", "Kocaeli", "Konya", 
            "Kütahya", "Malatya", "Manisa", "Kahramanmaraş", "Mardin", "Muğla", 
            "Muş", "Nevşehir", "Niğde", "Ordu", "Rize", "Sakarya", 
            "Samsun", "Siirt", "Sinop", "Sivas", "Tekirdağ", "Tokat", 
            "Trabzon", "Tunceli", "Şanlıurfa", "Uşak", "Van", "Yozgat", 
            "Zonguldak", "Aksaray", "Bayburt", "Karaman", "Kırıkkale", "Batman", 
            "Şırnak", "Bartın", "Ardahan", "Iğdır", "Yalova", "Karabük", 
            "Kilis", "Osmaniye", "Düzce"
        ]
        
        self.vector_store = None  # Will hold the FAISS vector store
        self.qa_chain = None      # Will hold the RetrievalQA chain
        self.sub_question_parser = PydanticOutputParser(pydantic_object=SubQuestionList)
        
        # Prompt template for generating sub-questions from a main question
        self.sub_question_prompt = PromptTemplate(
            template="""
            Aşağıdaki soruyu analiz et ve bu soruyu cevaplayabilmek için gerekli alt soruları oluştur.
            Her alt soru spesifik ve Wikipedia'da aranabilir olmalı.
            
            Ana Soru: {question}
            
            {format_instructions}
            
            Alt Sorular:
            """,
            input_variables=["question"],
            partial_variables={"format_instructions": self.sub_question_parser.get_format_instructions()}
        )
        
        # LLMChain for generating sub-questions using the LLM and the above prompt
        self.sub_question_chain = LLMChain(
            llm=self.llm,
            prompt=self.sub_question_prompt,
            output_parser=self.sub_question_parser
        )

    def fetch_wikipedia_content(self, city_name: str) -> str:
        """
        Fetches city information from Turkish Wikipedia. If the page exists, returns a summary and the first 5000 characters of the content.
        If not, tries an alternative search using the wikipedia package. Returns a message if content is not found.
        This function is robust to missing pages and will log errors if Wikipedia is unreachable or the city is not found.
        """
        try:
            # Try to fetch content using wikipediaapi (preferred for structured access)
            page = self.wiki_wiki.page(city_name)
            
            if page.exists():
                content = f"Şehir: {city_name}\n\n"
                content += f"Özet: {page.summary}\n\n"
                content += f"Tam İçerik: {page.text[:5000]}"  # Only first 5000 characters for efficiency
                return content
            else:
                # If not found, try using the wikipedia package as a fallback (less structured)
                wikipedia.set_lang("tr")
                try:
                    page_title = wikipedia.search(city_name, results=1)[0]
                    page_content = wikipedia.page(page_title)
                    content = f"Şehir: {city_name}\n\n"
                    content += f"Özet: {page_content.summary}\n\n"
                    content += f"Tam İçerik: {page_content.content[:5000]}"
                    return content
                except:
                    return f"Şehir: {city_name}\n\nİçerik bulunamadı." # If all fails, return a not found message
                    
        except Exception as e:
               # Log any unexpected errors during Wikipedia fetch
            logger.error(f"Wikipedia içeriği çekilemedi {city_name}: {str(e)}")
            return f"Şehir: {city_name}\n\nHata: İçerik çekilemedi."

    def build_vector_store(self):
        """
        Loads city information into the vector database. This method fetches Wikipedia content for each city,
        splits the text into chunks, and creates a FAISS vector store for semantic search and retrieval.
        This enables fast and relevant information retrieval for user questions.
        """
        logger.info("Şehir bilgileri yükleniyor...")
        
        documents = []  # Will hold Document objects for each city
        
        # Use ThreadPoolExecutor for parallel data fetching to speed up the process
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit fetch tasks for all cities
            futures = {executor.submit(self.fetch_wikipedia_content, city): city 
                      for city in self.turkish_cities}
            
            for future in futures:
                try:
                    content = future.result(timeout=30)
                    city = futures[future]
                    
                    # Only add documents if content was successfully fetched
                    if content and "İçerik bulunamadı" not in content:
                        doc = Document(
                            page_content=content,
                            metadata={
                                "city": city,
                                "source": "wikipedia",
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                        documents.append(doc)
                        logger.info(f"✓ {city} yüklendi")
                    else:
                        logger.warning(f"✗ {city} yüklenemedi")
                        
                except Exception as e:
                    logger.error(f"Hata: {futures[future]} - {str(e)}")
        
        if not documents:
            # If no city data could be loaded, raise an error to prevent further processing
            raise ValueError("Hiç şehir bilgisi yüklenemedi!")
        
        # Split the documents into smaller text chunks for better retrieval and semantic search
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Toplam {len(texts)} metin parçası oluşturuldu")
        
        # Create the FAISS vector store from the text chunks and embeddings
        self.vector_store = FAISS.from_documents(texts, self.embeddings)
        logger.info("Vektör veritabanı oluşturuldu")
        
        # Build the RetrievalQA chain for question answering using the vector store and LLM
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            return_source_documents=True
        )
        
        logger.info("RAG sistemi hazır!")

    def generate_sub_questions(self, question: str) -> List[str]:
        """
        Generates sub-questions from a main question using the LLM chain.
        This helps to break down complex queries into more manageable and answerable parts.
        If an error occurs, returns the main question as a fallback to ensure the process continues.
        """
        try:
            result = self.sub_question_chain.run(question=question)
            return [sq.question for sq in result.sub_questions]
        except Exception as e:
            logger.error(f"Alt soru üretme hatası: {str(e)}")
            return [question]  # Fallback: return the main question

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answers the user's question in detail. This method generates sub-questions, retrieves answers for each,
        and then provides a final answer using the QA chain. Also returns source cities and OpenAI API cost info.
        This is the main entry point for answering user queries via the web interface.
        """
        if not self.qa_chain:
            # If the QA chain is not ready, return an error
            return {"error": "RAG sistemi henüz hazır değil"}
        
        try:
            with get_openai_callback() as cb:
                # Generate sub-questions for the main question
                sub_questions = self.generate_sub_questions(question)
                
                # Retrieve answers for each sub-question using the QA chain
                sub_answers = []
                for sub_q in sub_questions:
                    sub_result = self.qa_chain({"query": sub_q})
                    sub_answers.append({
                        "question": sub_q,
                        "answer": sub_result["result"],
                        "sources": [doc.metadata.get("city", "Unknown") 
                                  for doc in sub_result["source_documents"]]
                    })
                
                # Retrieve the final answer for the main question
                final_result = self.qa_chain({"query": question})
                
                return {
                    "question": question,
                    "answer": final_result["result"],
                    "sub_questions": sub_answers,
                    "source_cities": list(set([doc.metadata.get("city", "Bilinmeyen") 
                                            for doc in final_result["source_documents"]])),
                    "cost": {
                        "total_tokens": cb.total_tokens,
                        "total_cost": cb.total_cost
                    }
                }
                
        except Exception as e:
            # Log any errors that occur during the answer process
            logger.error(f"Soru cevaplama hatası: {str(e)}")
            return {"error": f"Cevap üretilirken hata: {str(e)}"}

# Global RAG sistemi
rag_system = None

def initialize_rag_system():
    """RAG sistemini başlatır"""
    global rag_system
    
    try:
        rag_system = TurkishCityRAG()
        rag_system.build_vector_store()
        return True
    except Exception as e:
        logger.error(f"RAG sistem başlatma hatası: {str(e)}")
        return False

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    global rag_system
    
    if not rag_system:
        return jsonify({"error": "RAG sistemi henüz hazır değil. Lütfen biraz bekleyin."})
    
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({"error": "Lütfen bir soru yazın"})
        
        # Soruyu cevapla
        result = rag_system.answer_question(question)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API hatası: {str(e)}")
        return jsonify({"error": f"Sunucu hatası: {str(e)}"})

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "rag_system_ready": rag_system is not None,
        "cities_count": len(rag_system.turkish_cities) if rag_system else 0
    })

if __name__ == '__main__':
    print("🚀 Türkiye Şehirleri RAG Sistemi Başlatılıyor...")
    print("📝 OPENAI_API_KEY environment variable'ını set etmeyi unutmayın!")
    
    # RAG sistemini arka planda başlat
    import threading
    
    def init_rag():
        if initialize_rag_system():
            print("✅ RAG sistemi hazır!")
        else:
            print("❌ RAG sistemi başlatılamadı!")
    
    # Arka planda RAG sistemini başlat
    thread = threading.Thread(target=init_rag)
    thread.daemon = True
    thread.start()
    
    # Flask uygulamasını başlat
    app.run(debug=True, host='0.0.0.0', port=5000)
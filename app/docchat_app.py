import os
import shutil
from typing import Optional
import requests
from typing import List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi import File, UploadFile
from pydantic import BaseModel

from haystack.nodes import Crawler
from haystack.nodes import PDFToTextConverter
from haystack.utils import convert_files_to_docs
from haystack.nodes import PreProcessor

# from haystack.document_stores import MilvusDocumentStore
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever
from haystack.utils import convert_files_to_docs, clean_wiki_text
from haystack.nodes import Seq2SeqGenerator
from haystack.pipelines import GenerativeQAPipeline
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents, print_answers

app = FastAPI()

DOCS_DIR = 'docs'
TXT_DIR = 'TXT'
LINK_DIR = 'links'
PROCESSED_DOCS = 'processed'
SQL_FILE = 'faiss_doc_store.db'
FAISS_FILE = 'faiss_index.faiss'

for _dir in [DOCS_DIR, TXT_DIR, LINK_DIR, PROCESSED_DOCS]:
    if not os.path.exists(_dir):
        os.makedirs(_dir)


class Url(BaseModel):
    title: Optional[str] = None
    url: str


class AiQA():
    # Create a Singleton class
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self):
        # cls.document_store = MilvusDocumentStore(host='192.168.1.17',
        #                             embedding_dim=128,
        #                             duplicate_documents='overwrite',
        #                             recreate_index=False)

        if os.path.exists(SQL_FILE) and os.path.exists('my_faiss'):
            self.document_store = FAISSDocumentStore.load(index_path="my_faiss")
        # if os.path.exists(SQL_FILE):
        #     os.remove(SQL_FILE)
        else:
            self.document_store = FAISSDocumentStore(embedding_dim=128,
                                        faiss_index_factory_str="Flat",
                                        sql_url=f"sqlite:///{SQL_FILE}")

        # self.document_store = FAISSDocumentStore(embedding_dim=128,
        #                                          faiss_index_factory_str="Flat",
        #                                          sql_url=f"sqlite://")

        self.retriever = None
        self.generator = None


    # @classmethod
    # def create_index_milvus(cls, docs):
    #     cls.document_store.write_documents(docs)

    #     cls.retriever = DensePassageRetriever(
    #         document_store=cls.document_store,
    #         query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
    #         passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
    #         use_gpu=True
    #     )

    #     cls.document_store.update_embeddings(cls.retriever)


    def create_index_faiss(self, docs):
        # cls.document_store= FAISSDocumentStore.load(faiss_path)

        self.document_store.write_documents(docs)

        # %% Initialize Retriever and Reader/Generator

        # Retriever (DPR)
        if self.retriever == None:
            self.retriever = DensePassageRetriever(
                document_store=self.document_store,
                query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
                passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
                use_gpu=False
            )

        self.document_store.update_embeddings(retriever=self.retriever,
            update_existing_embeddings=False)

        self.document_store.save("my_faiss")

    def answer(self, question):
        if self.generator is None:
            self.generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa",
                                use_gpu=True)
            self.pipe = GenerativeQAPipeline(self.generator, self.retriever)
        k_retriever = 3
        self.res = self.pipe.run(question, params={"Retriever": {"top_k": k_retriever}})

        print_answers(self.res, details="minimum")


aiqa = AiQA()


@app.get('/')
async def root(request: Request):
    # local ip: <a href={request.url._url}docs>{request.url._url}docs/</a>
    ip = requests.get('https://api.ipify.org').content.decode('utf8')
    html_content = f"""
    <html>
        <head>
            <title>Document Store</title>
        </head>
        <body><center>
        <br/>
            <h1>֍ Welcome to the Intelligent Document Retrieval API ֍</h1>
            <h2>► Here is the root URL</h2>
            <ul style="display:inline;">
                <li> • Visit <strong>
                    <a href=/docs>{ip}/docs/</a>
                </strong> to view and test API endpoints</li>
            </ul>
        </body></center>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/url")
async def get_url(link: Url):
    crawler = Crawler(output_dir=LINK_DIR, crawler_depth=1)
    crawled_docs = crawler.crawl(urls=[link.url])
    return f"Started processing {link.url}"


@app.post("/doc")
def upload(file: UploadFile = File(...)):
    try:
        out_path = os.path.join(DOCS_DIR, file.filename)
        if os.path.exists(out_path):
            return {"message": "Uploaded filename already exists in our docs!"}
        contents = file.file.read()
        with open(out_path, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    return {"message": f"Successfully Uploaded {file.filename}"}


@app.get("/doc/process")
def process_docs():    
    # %% All doc types
    all_docs = convert_files_to_docs(dir_path=DOCS_DIR)

    for doc_txt in all_docs:
        doc_filename = f"{os.path.basename(doc_txt.meta['name']).split('.')[0]}.txt"
        with open(os.path.join(TXT_DIR, doc_filename), 'w') as f:
            f.write(doc_txt.content)

    # %% Preprocess
    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=False,
        split_by="word",
        split_length=200,
        split_overlap=50,
        split_respect_sentence_boundary=True,
    )

    docs_default = preprocessor.process(all_docs)
    print(f"n_docs_input: {len(all_docs)}\nn_docs_output: {len(docs_default)}")

    for chunk in docs_default:
        chunk_filename = f"{chunk.meta['name']}_{chunk.meta['_split_id']}.txt"
        with open(os.path.join(PROCESSED_DOCS, chunk_filename), 'w') as f:
            f.write(chunk.content)

    return {"message": f"Successfully processed {len(all_docs)} document(s) and created {len(docs_default)} passages"}


@app.post('/doc/get_docs')
async def get_docs(files: List[UploadFile]):
    for file in files:
        contents = await file.read()
        with open(os.path.join(PROCESSED_DOCS, file.filename), 'wb') as f:
            f.write(contents)

    index_documents()
    # return {"filenames": [file.filename for file in files]}
    return {f"Received and indexed {len(files)} documents"}


@app.get("/ai/index")
def index_documents():
    aiqa = AiQA()
    print("AiQA:", aiqa)
    # # Convert files to docs + cleaning
    docs = convert_files_to_docs(dir_path=PROCESSED_DOCS,
                                clean_func=clean_wiki_text,
                                split_paragraphs=True)

    # AiQA.create_index_milvus(docs)
    aiqa.create_index_faiss(docs)

    # %% ----------------MILVUS----------------
    # document_store = MilvusDocumentStore(host='192.168.1.17',
    #                                 embedding_dim=128,
    #                                 duplicate_documents='overwrite',
    #                                 recreate_index=False)

    # document_store.write_documents(docs)

    # retriever = DensePassageRetriever(
    #     document_store=document_store,
    #     query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
    #     passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
    #     use_gpu=True
    # )

    # document_store.update_embeddings(retriever)

    

    # # Test DPR
    # p_retrieval = DocumentSearchPipeline(AiQA.retriever)
    # res = p_retrieval.run(query="Tell me something about global greenhouse gas emissions?", params={"Retriever": {"top_k": 5}})
    # print_documents(res, max_text_len=512)

    if os.path.exists(PROCESSED_DOCS):
        shutil.rmtree(PROCESSED_DOCS)
    if not os.path.exists(PROCESSED_DOCS):
        os.makedirs(PROCESSED_DOCS)

    return {"message": f"Successfully indexed processed passages"}


@app.get('/ai/answer/{question}')
def answer(question: str):
    aiqa = AiQA()
    # %% Generator
    # generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa",
    #                             use_gpu=True)

    # pipe = GenerativeQAPipeline(generator, AiQA.retriever)

    # k_retriever = 3
    # res = pipe.run(question, params={"Retriever": {"top_k": k_retriever}})

    # print_answers(res, details="minimum")
    if aiqa.retriever == None:
        aiqa.retriever = DensePassageRetriever(
                document_store=aiqa.document_store,
                query_embedding_model="vblagoje/dpr-question_encoder-single-lfqa-wiki",
                passage_embedding_model="vblagoje/dpr-ctx_encoder-single-lfqa-wiki",
                use_gpu=False
        )

    aiqa.document_store.update_embeddings(retriever=aiqa.retriever,
                                          update_existing_embeddings=False)
    aiqa.answer(question)
    for i in range(3):
        print(aiqa.res['answers'][0].meta['content'][i])
    return {'answer': aiqa.res['answers'][0].answer}


if __name__ == '__main__':
    import uvicorn
    # uvicorn.run(app)
    uvicorn.run(app, port=8080, host='0.0.0.0')
    
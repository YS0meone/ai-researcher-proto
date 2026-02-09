# # # understand 
# from app.services.s2_client import S2Client
# from collections import defaultdict
# from langchain_openai import OpenAIEmbeddings

# from pprint import pprint
# from langchain_community.document_loaders.generic import GenericLoader
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders.parsers import GrobidParser
# from langchain_classic.retrievers import ParentDocumentRetriever
# from langchain_core.stores import InMemoryByteStore
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# import os
# from qdrant_client.models import VectorParams
# from langchain_core.documents import Document
# from app.core.config import settings
# from app.agent.RedisDocumentStore import RedisDocumentStore


# store = RedisDocumentStore(redis_url="redis://localhost:6379")
# os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
# # store = RedisStore(redis_url="redis://localhost:6379")
# # s2_client = S2Client()

# # results = s2_client.search_papers(query="paperqa")
# # # pprint(results[0].model_dump())


# # import matplotlib.pyplot as plt
# # import arxiv
# # import re
# # client = arxiv.Client(
# #     delay_seconds=3.0,  # Wait 3 seconds between requests
# #     num_retries=3)
# # #         # search for the most recent 100 papers in the cs.LG category
# # search = arxiv.Search(
# #     query=f'ti:"{results[0].title}"',
# #     max_results=1,
# #     sort_by=arxiv.SortCriterion.Relevance,
# #     sort_order=arxiv.SortOrder.Descending
# #     )
# # arxiv_paper = None
# # for result in client.results(search):
# #     arxiv_paper = result
# #     break

# # file_name = re.sub(r'[<>:"/\\|?*]', '_', arxiv_paper.title.replace(" ", "_"))
# # arxiv_paper.download_pdf(dirpath=".", filename=file_name+".pdf")

# def add_s2_paper(file_name: str, folder_path: str):
#     loader = GenericLoader.from_filesystem(
#         folder_path,
#         glob=file_name+".pdf",
#         suffixes=[".pdf"],
#         parser= GrobidParser(segment_sentences=False)
#     )
#     docs = loader.load()

#     # section_counter = defaultdict(int)
#     # section_word_count = defaultdict(int)
#     # for doc in docs:
#     #     if doc.metadata.get("section_title", None):
#     #         section_counter[doc.metadata.get("section_title")] += 1
#     #         section_word_count[doc.metadata.get("section_title")] += len(doc.page_content.split())
#     # pprint(section_counter)
#     # pprint(section_word_count)
#     # print(len(section_counter))
#     # print(len(docs))

#     parent_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=2500,
#     chunk_overlap=250,
#     add_start_index=True
# )

#     # Child chunks: ~100 words (500 chars) - good for precise retrieval
#     child_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         add_start_index=True
#     )

#     # pprint(docs[0].metadata)
#     # for doc in range(min(20, len(docs))):
#     #     print(docs[doc].page_content)
#     #     print("--------------------------------")
    
#     client=QdrantClient(url="http://localhost:6333")
#     client.delete_collection(collection_name="papers")
#     client.create_collection(
#                 collection_name="papers",
#                 vectors_config=VectorParams(
#                     size=1536,
#                     distance="Cosine",
#                 ),
#             )
#     vectorstore = QdrantVectorStore(
#         client=client,
#         collection_name="papers",
#         embedding = OpenAIEmbeddings(model="text-embedding-3-small") 
#     )

#     retriever = ParentDocumentRetriever(
#         vectorstore=vectorstore,
#         docstore=store,
#         child_splitter=child_splitter
#     )
    
#     section_docs = defaultdict(list)
#     for doc in docs:
#         section_docs[doc.metadata.get("section_title", "General")].append(doc)
#     new_docs = []
#     for docs in section_docs.values():
#         metadata = docs[0].metadata.copy()
#         del metadata["text"]
#         del metadata["file_path"]
#         del metadata["bboxes"]
#         new_docs.append(Document(page_content="\n\n".join([doc.page_content for doc in docs]), metadata=metadata))
    
#     # print(len(new_docs))
#     # print(new_docs[0].metadata)
#     # print(new_docs[0].page_content)

#     retriever.add_documents(new_docs)

#     results = retriever.invoke("What is paperqa system?")

#     print(results)

# add_s2_paper("PaperQA__Retrieval-Augmented_Generative_Agent_for_Scientific_Research", ".")

from app.services.qdrant import QdrantService
from app.core.config import settings
from app.db.schema import S2Paper

qdrant_service = QdrantService(settings.qdrant_config)

print(qdrant_service.add_s2_paper_abstract_only(S2Paper(paperId="2508.11957v1", title="Test Paper", abstract="This is a test abstract")))
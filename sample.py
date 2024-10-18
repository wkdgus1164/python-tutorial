from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI


def calculate(query: str = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'):
    load_dotenv()

    text_splitter = RecursiveCharacterTextSplitter(

        # 문서를 쪼갤 때 얼마나 많은 토큰 (단어) 수마다 자를지 결정
        chunk_size=1500,

        # 문맥을 파악하기 위해 다음 자름 포인트보다 overlap 만큼 더 앞 부분부터 자름
        chunk_overlap=200,
    )

    loader = Docx2txtLoader('./tax.docx')

    # 이렇게 가져오면 통 리스트 하나로 가져올 수 있음
    # document = loader.load()

    # 이렇게 가져오면 `text_splitter` 로 자른 사이즈들이 리스트데 담겨서 가져와짐
    document_list = loader.load_and_split(text_splitter=text_splitter)

    # 이렇게 불러오면 하나의 리스트 형태로 불러와진다. (length 1)
    # 이걸 쪼개야 하는데, 여기서 text-splitter 가 필요하다.
    # pip install -qU langchain-text-splitters

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    # 벡터 데이터베이스인 Chroma 를 사용한다.
    # %pip install langchain-chroma

    # 위에서 쪼개놓은 문서를 기반으로 백터 데이터베이스에 저장한다.
    database = Chroma.from_documents(

        # 문서
        documents=document_list,

        # 임베드 객체
        embedding=embedding,

        # 히스토리를 저장할 DB
        persist_directory='./Chroma',

        # DB Name
        collection_name='chroma-tax'
    )

    # query = '연봉 5천만원인 직장인의 소득세는 얼마인가요?'

    # 백터 DB 에서 유사도 검색을 한다.
    # 위에 넣었던 임베딩을 활용해서 알아서 답변을 가져온다.
    retrieved_docs = database.similarity_search(query)

    llm = ChatOpenAI(model='gpt-4o')

    promt = f"""[Identity]
        - 당신은 최고의 한국 소득세 전문가입니다.
        - [Context]를 참고해서 사용자의 질문에 답변해주세요.
        
        [Context]
        {retrieved_docs}
        
        Question: {query}
        """

    ai_message = llm.invoke(promt)

    return ai_message.content

print(calculate())
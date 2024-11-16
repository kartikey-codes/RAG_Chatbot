import streamlit as st
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import PyPDF2
import io
from docx import Document as DocxDocument


# Initialize Together AI model
# Access secrets from Streamlit Secrets Manager
api_key = st.secrets["together_ai"]["api_key"]
model = st.secrets["together_ai"]["model"]

# Initialize Together AI model with secrets
chat_model = ChatTogether(
    together_api_key=api_key,
    model=model
)

# Prompt engineering
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert assistant. Use the provided context to answer questions accurately and concisely.\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer (be specific and avoid hallucinations):"
    )
)

# Define the QA chain
qa_chain = LLMChain(llm=chat_model, prompt=prompt_template)

st.title("📝 File Q&A")
uploaded_files = st.file_uploader("Upload articles", type=("txt", "pdf", "docx"), accept_multiple_files=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Upload documents and ask questions about them!"}]
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = set()
if "file_contents" not in st.session_state:
    st.session_state["file_contents"] = {}

# Disable the chat input until at least one file is uploaded
chat_disabled = len(uploaded_files) == 0

# File processing
for uploaded_file in uploaded_files:
    if uploaded_file.name not in st.session_state["file_contents"]:
        try:
            # Handle different file types
            if uploaded_file.type == "text/plain":
                file_content = uploaded_file.getvalue().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                file_content = ""
                for page in pdf_reader.pages:
                    file_content += page.extract_text()
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                doc = DocxDocument(io.BytesIO(uploaded_file.getvalue()))
                file_content = ""
                for para in doc.paragraphs:
                    file_content += para.text + "\n"
            else:
                st.error(f"Unsupported file type for {uploaded_file.name}. Please upload .txt, .pdf, or .docx files.")
                continue

            # Split the content into chunks
            document = Document(page_content=file_content, metadata={"source": uploaded_file.name})
            chunk_size = 1000
            chunk_overlap = 200

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = text_splitter.split_documents([document])

            st.session_state["file_contents"][uploaded_file.name] = "\n\n".join([chunk.page_content for chunk in chunks])
            st.success(f"Processed {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Toggleable chat history
with st.expander("Chat History"):
    for msg in st.session_state.messages:
        if msg["role"] == "assistant":
            # Assistant's message on the left
            st.markdown(f"<div style='text-align: left; background-color: #f4f4f9; padding: 10px; border-radius: 10px; width: 80%; margin: 5px;'>Assistant: {msg['content']}</div>", unsafe_allow_html=True)
        else:
            # User's message on the right
            st.markdown(f"<div style='text-align: right; background-color: #d1e7fd; padding: 10px; border-radius: 10px; width: 80%; margin: 5px;'>User: {msg['content']}</div>", unsafe_allow_html=True)

# Chat functionality (send button for user input)
question = st.text_input("Ask a question about the uploaded documents:", disabled=chat_disabled)

if st.button("Send"):
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            try:
                # Combine contents from all uploaded files
                all_contents = "\n\n".join(
                    [f"Content of {filename}:\n{content}" for filename, content in st.session_state["file_contents"].items()]
                )

                # Use the QA chain to get an answer
                answer = qa_chain.run(context=all_contents, question=question)

                # Display the assistant's response
                st.write(answer)

                # Save the assistant's response in session state
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error querying Together AI model: {str(e)}")

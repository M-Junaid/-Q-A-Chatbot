#Langchain_api_key =" ls__c3be3ee182d840778c6a65f7411fb528"
#Langchain_Project = "my_project"
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from decouple import config
HUGGINGFACEHUB_API_TOKEN = config("Huggingfacehub_api_key")

templete = "<s>[INST]write short answer of </s>{question}[/INST]"
print(templete)

prompt_template = PromptTemplate.from_template(templete)
fromatted_prompt_template = prompt_template.format(
    question = "who is priministor of pakistan"
)

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm= HuggingFaceEndpoint(
    repo_id=repo_id,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

response = llm.invoke(fromatted_prompt_template)
print(response)
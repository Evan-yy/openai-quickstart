import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def initialize_sales_bot(vector_store_dir: str="real_estates_sale"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(base_url="https://api.xiaoai.plus/v1",api_key="sk-yUiGr5Uis41lNvLc5a909249B1A945DaAaF197D032A342A6"),allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, base_url="https://api.xiaoai.plus/v1",api_key="sk-yUiGr5Uis41lNvLc5a909249B1A945DaAaF197D032A342A6")
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    formatted_history = "\n".join([f"客戶: {h[0]}\n銷售助理: {h[1]}" for h in history])
    ans = SALES_BOT({"query": message, "chat_history": formatted_history})
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
      if len(ans["source_documents"]) == 0:
            template = """
            你是一个专业的的车辆销售助理，你的回答应该足够自然、友好，要站在这个岗位的角度回答客户，并且请你记住以下几点：
                1. 用自然的语气交谈，就像真人一样，避免机械式的回答。
                2. 如果不确定，可以做出合理的假设，并且在答案中提现出来。
                3. 不要提及你是一个大模型或者是人工智能，而要代入到你是一个专业的汽车店销售助理
            以下是之前的对话：
            {history}
            客戶的最新回答是：{question}
            请给出一个自然、连贯的回复，要像真人销售一样：
            """
            llm = ChatOpenAI(model_name="gpt-4", temperature=0, base_url="https://api.xiaoai.plus/v1",api_key="sk-yUiGr5Uis41lNvLc5a909249B1A945DaAaF197D032A342A6")

            prompt = PromptTemplate(template=template, input_variables=["history", "question"])
            chain = LLMChain(llm=llm, prompt=prompt)

            response = chain.run(history=formatted_history, question=message)

            return response    
        else:
            return ans["result"]
    else:
        return "这个问题我要问问领导"
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="房产销售",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()

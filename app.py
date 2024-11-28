from openai import AzureOpenAI
from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential,get_bearer_token_provider
from numpy import dot
from numpy.linalg import norm
import gradio as gr
import time
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify, render_template

load_dotenv()

# Acquire a credential object
azureCredential = DefaultAzureCredential()

# Set up Azure authentication
token_provider = get_bearer_token_provider(
    azureCredential, "https://cognitiveservices.azure.com/.default"
)

# Set your OpenAI and Azure Cosmos credentials
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
use_apim = os.getenv("USE_APIM", "false").lower() == "true"

if use_apim:
    azOpenAIClient = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )
else:
    azOpenAIClient = AzureOpenAI(
        azure_endpoint=azure_endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version
    )

cosmos_url = os.getenv("AZURE_COSMOSDB_URL")

# Initialize Cosmos DB client
client = CosmosClient(cosmos_url, credential=azureCredential)
database = client.get_database_client("ProjectRiskDB")
container = database.get_container_client("RisksAndProjects")

# Helper function: Calculate cosine similarity
def cosine_similarity(vector1, vector2):
    return dot(vector1, vector2) / (norm(vector1) * norm(vector2))

# Helper function: Retrieve relevant risks from Cosmos DB
def retrieve_relevant_risks(project_description, top_k_projects=1):
    # Generate embedding for project description
    embedding_response = azOpenAIClient.embeddings.create(
        input=project_description,
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    )
    embeddings = embedding_response.model_dump() 
    project_embedding = embeddings["data"][0]["embedding"]

    project_summaries = container.query_items(
        query="SELECT c.id, c.project_name, c.embedding FROM c WHERE c.type = 'project_summary'",
        enable_cross_partition_query=True
    )

    # Step 3: Find the most similar project(s) based on embeddings
    project_similarities = []
    for project in project_summaries:
        similarity = cosine_similarity(project_embedding, project["embedding"])
        project_similarities.append((project["id"], similarity, project["project_name"]))

    # Sort by similarity and pick the top matching project(s)
    project_similarities.sort(key=lambda x: x[1], reverse=True)
    top_projects = project_similarities[:top_k_projects]

    if not top_projects:
        raise ValueError("No similar projects found.")

    risks = []
    for project_id, _, project_name in top_projects:
        project_risks = container.query_items(
            query="SELECT * FROM c WHERE c.type = 'risk' AND c.project_name = @project_name",
            parameters=[{"name": "@project_name", "value": project_name}],
            enable_cross_partition_query=True
        )
        risks.extend(project_risks)

    return risks, top_projects

# Helper function: Format augmented input for AI
def format_augmented_input(project_description, relevant_risks, matched_projects):
    input_text = f"Project Description:\n{project_description}\n\nRelevant Risks:\n"

    # Include matched projects as reference
    input_text += "Matched Projects (used as reference):\n"
    for idx, (project_id, similarity, project_name) in enumerate(matched_projects):
        input_text += f"Project {idx + 1}: {project_name} (Similarity Score: {similarity:.2f})\n"


    for idx, (risk_id, similarity, risk_data) in enumerate(relevant_risks):
        input_text += f"\nRisk {idx + 1}:\n"
        input_text += f"Name: {risk_data['risk_name']}\n"  # Access risk_name directly from document
        input_text += f"Description: {risk_data['description']}\n"  # Access description directly
        input_text += f"Likelihood: {risk_data['metadata']['qualitative_scoring']['Current Likelihood Rating']}\n"
        input_text += f"Impact: {risk_data['metadata']['qualitative_scoring']['Current Rating']}\n"
        input_text += f"Controls: {', '.join(risk_data['metadata']['controls'])}\n"
    return input_text

# Main function: Generate a risk profile
def generate_risk_profile(project_description):
    # Step 1: Retrieve relevant risks
    relevant_risks, matched_projects = retrieve_relevant_risks(project_description)
    # Step 2: Augment the input
    augmented_input = format_augmented_input(project_description, relevant_risks, matched_projects)

    # Step 3: Generate risk profile using OpenAI
    messages = [
        {"role": "system", "content": "You are an expert risk management consultant. Rely only on the context provided to generate the risk profile, and don't make up advice."},
        {"role": "user", "content": f"""
         Based on the following context, generate a risk profile for the project:\n\n{augmented_input}. 

         Outline the risk profile in the following format:
            1. List the Top 3 risks by value for this kind of project.
            2. Tabulate all risks with all relevant columns including the best case, most likely, worst case $ estimates and the control and treatment actions.
            3. Information used as basis for the generation of this risk profile - summarised
         """}
    ]

    response = azOpenAIClient.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_GPT_DEPLOYMENT_NAME"),
        messages=messages,
        max_tokens=1000,
        temperature=0.7
    )

    
    completion_response = response.model_dump()
    print("total tokens = ", completion_response["usage"]["total_tokens"])

    return completion_response["choices"][0]["message"]["content"]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    start_time = time.time()
    response_payload = generate_risk_profile(user_message)
    end_time = time.time()
    elapsed_time = round((end_time - start_time) * 1000, 2)
    details = f"\n (Time: {elapsed_time}ms)"
    response = {
        "user_message": user_message,
        "response": response_payload + details
    }
    return jsonify(response)

# Example usage
if __name__ == "__main__":
    app.run(host='0.0.0.0')


    # # Example project description
    # planned_project_description = """
    # Design and commissioning of new rolling stock trains for a metropolitan transit system. The project involves tight timelines, multiple stakeholders, and unionized labor.
    # """

    # chat_history = []

    # app = FastAPI()

    # with gr.Blocks(fill_height=True, fill_width=True) as demo:
    #     chatbot = gr.Chatbot(label="AI PM Assistant", scale=5)
    #     msg = gr.Textbox(label="Ask me about generating risk profiles for Projects!", scale=1)
    #     clear = gr.Button("Clear", scale=0)

    #     def user(user_message, chat_history):
    #         start_time = time.time()
    #         response_payload = generate_risk_profile(user_message)
    #         end_time = time.time()
    #         elapsed_time = round((end_time - start_time) * 1000, 2)
    #         details = f"\n (Time: {elapsed_time}ms)"
    #         chat_history.append([user_message, response_payload + details])
            
    #         return gr.update(value=""), chat_history
        
    #     msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    #     clear.click(lambda: None, None, chatbot, queue=False)

    # app = gr.mount_gradio_app(app, demo, path="/")

    # # Launch the Gradio interface
    # demo.launch(debug=True)

    # # Be sure to run this cell to close or restart the Gradio demo
    # demo.close()
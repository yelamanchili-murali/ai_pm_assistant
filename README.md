# Project Title

## Azure Prerequisites

Before running this application, ensure you have the following Azure resources and credentials:

1. **Azure OpenAI Service**:
    - Endpoint URL
    - API Version
    - Deployment names for embedding and GPT models

2. **Azure Cosmos DB**:
    - Cosmos DB URL

3. **API Key**:
    - API key for Azure OpenAI or Azure API Management (APIM) subscription key

## Setup Instructions

### Step 1: Create a .env File

1. Copy the `.env.sample` file to create a new `.env` file:
    ```sh
    cp .env.sample .env
    ```

2. Open the `.env` file and replace the placeholder values with your actual credentials and configuration.

### Step 2: Create Azure Cosmos DB and Container

1. Go to the [Azure Portal](https://portal.azure.com/).

2. Create a new Azure Cosmos DB account if you don't have one:
    - Select **Azure Cosmos DB** and click **Add**.
    - Choose **Core (SQL) - Recommended** as the API.
    - Fill in the required details and create the account.

3. Create a new database and container:
    - Navigate to your Cosmos DB account.
    - Click on **Data Explorer**.
    - Click **New Database** and enter the database ID as `ProjectRiskDB`.
    - Click **New Container** and enter the following details:
        - Database ID: `ProjectRiskDB`
        - Container ID: `RisksAndProjects`
        - Partition Key: `/type`

4. Add documents to the container:
    - Navigate to the `RisksAndProjects` container.
    - Click **Items** and then **New Item**.
    - Add the documents from the `cosmos_documents` folder as items in the container.

### Step 3: Install Dependencies

Make sure you have Python installed. Then, install the required dependencies using pip:
```sh
pip install -r requirements.txt
```

#### Step 4: Run the Application

Run the Python script:

```sh
python generate_risk_profile.py
```

This will execute the script using the configuration specified in your .env file.

Replace the placeholder values in the .env file with your actual values before running the application.
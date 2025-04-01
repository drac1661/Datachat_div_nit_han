import os
import sqlite3
import pandas as pd
from pydantic import BaseModel
import io
import requests
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    APIRouter,
    Request,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Dict, Union
from datetime import timedelta
import json
from starlette.middleware.sessions import SessionMiddleware
from starlette.config import Config
from starlette.responses import RedirectResponse
from uuid import uuid4
from fastapi.responses import JSONResponse

load_dotenv()

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    session_cookie="session",  # Name for session cookies
    secret_key="Drac",  # Similar to SESSION_COOKIE_SAMESITE
)
router = APIRouter(prefix="/datachat")


# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def get_session(request: Request):
    return request.session


class QuestionRequest(BaseModel):
    question: str


class QueryRequest(BaseModel):
    query: str


class FormatResponseRequest(BaseModel):
    question: str
    result: Union[str, List[Dict]]
    query: str


class ReGenerateRequest(BaseModel):
    question: str
    previous_sql_query: str
    previous_explanation: str
    user_description: str


# Modify the global SQLite connection to be session-specific
def get_db_connection(session_id: str):
    """Creates or retrieves a session-specific SQLite connection."""
    conn = sqlite3.connect(f"{session_id}.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@router.get("/start-session")
async def start_session():
    """Starts a new session and returns a session ID."""
    session_id = str(uuid4())
    response = JSONResponse(content={"session_id": session_id})
    response.set_cookie(
        key="session_id",
        value=session_id,
        path="/datachat",
        httponly=True,
        samesite="none",  # Required for cross-site cookies
        secure=True,
    )
    return response


@router.get("/healthcheck/")
async def healthcheck():
    """Healthcheck endpoint for the API."""
    return {"status": "API is running smoothly!"}, 200


@router.post("/upload-csv/")
async def upload_csv(request: Request, files: List[UploadFile] = File(...)):
    """Uploads multiple CSV files and stores them in session-specific SQLite tables."""
    session_id = request.cookies.get("session_id")
    for i in request.cookies:
        print(i)
    if not session_id:
        raise HTTPException(
            status_code=400, detail="Session ID not found. Start a session first."
        )

    try:
        conn = get_db_connection(session_id)
        cursor = conn.cursor()
        table_names = []
        for file in files:
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode("utf-8", errors="ignore")))

            if df.empty:
                raise HTTPException(
                    status_code=400, detail=f"Uploaded CSV {file.filename} is empty!"
                )

            # Ensure valid column names
            df.columns = [col.strip().replace(" ", "_") for col in df.columns]

            # Create a unique table name based on the file name
            table_name = f"{file.filename.split('.')[0].replace(' ', '_')}"
            table_names.append(table_name)

            # Store data in SQLite
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        return {"message": "CSV files uploaded successfully!", "tables": table_names}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


@router.get("/extract-schema/")
def extract_schema(session_id: str):
    """Extracts and returns the database schema for all tables in the user's session."""

    try:
        conn = get_db_connection(session_id)
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = cursor.fetchall()

        if not tables:
            return {"error": "No tables found. Please upload CSV files first."}

        schemas = []
        for table in tables:
            table_name = table[0]

            # Retrieve table creation statement
            cursor.execute(
                f'SELECT sql FROM sqlite_master WHERE type="table" AND name="{table_name}"'
            )
            create_table_sql = cursor.fetchone()[0]

            # Retrieve column details
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            column_details = [
                {
                    "Column Name": col[1],
                    "Type": col[2],
                    "Not Null": "Yes" if col[3] else "No",
                    "Default Value": col[4] if col[4] else "NULL",
                    "Primary Key": "Yes" if col[5] else "No",
                }
                for col in columns
            ]

            schema = {
                "table_name": table_name,
                "create_statement": create_table_sql,
                "columns": column_details,
            }
            schemas.append(schema)

        return {"schemas": schemas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-query/")
def generate_query(request: QuestionRequest):
    """Generates an SQL query and its explanation based on user input."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=400, detail="Session ID not found. Start a session first."
        )

    schema_response = extract_schema(session_id)
    if "error" in schema_response:
        raise HTTPException(status_code=400, detail=schema_response["error"])

    schemas = schema_response["schemas"]
    schema_texts = [
        f"Table '{schema['table_name']}':\n{schema['create_statement']}'"
        for schema in schemas
    ]
    schema_text = "\n\n".join(schema_texts)

    prompt = f"""
    You are an expert SQLite query writer. Here are the database schemas:
    {schema_text}

    User's Question: {request.question}

    Write a SQLite query to answer the question. The query should be optimized and syntactically correct, using joins if necessary. Ensure:

    - The query is strictly relevant to the question, without unnecessary columns or conditions.
    - It follows best practices for performance and readability.
    - If aggregation, filtering, or ordering is required based on the question, include it appropriately.
    - If any assumptions need to be made due to missing details, state them clearly before the query.
    - Provide only the final query unless additional context is required for clarification.
    
    *"Analyze and explain the following SQL query in a detailed yet easy-to-understand manner. Break down the thought process behind constructing the query, including the purpose of each clause, how different components interact, and why specific functions or joins are used. Structure the explanation logically with the following sections and return whole explanation in a single string but with proper line breaks and indentations:

    Query Overview – Summarize what the query aims to achieve.
    Step-by-Step Breakdown – Explain each part of the query in sequence, including SELECT, FROM, JOINs, WHERE, GROUP BY, HAVING, ORDER BY, and any other clauses.
    Logical Flow – Describe how data flows through the query, from filtering to aggregation and final output.
    Optimization Considerations – Discuss any performance aspects, such as indexing, subqueries vs. joins, or potential improvements.
    Alternative Approaches – If applicable, suggest different ways to achieve the same result.
    Ensure the explanation is thorough yet easy to follow, avoiding unnecessary complexity while maintaining technical accuracy."*
    Always use column names in brackets [] without altering the original column names in case of LIKE statements.
    Provide your response in the following JSON format:
    {{
        "sql_query": "<YOUR SQL QUERY HERE>",
        "explanation": "<YOUR EXPLANATION HERE>"
    }}
    """

    response = requests.post(
        "https://llmfoundry.straive.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('LLMFOUNDRY_TOKEN')}:my-test-project"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    response_json = response.json()

    if "choices" not in response_json or len(response_json["choices"]) == 0:
        raise HTTPException(status_code=500, detail="Invalid response from LLM.")

    llm_message = response_json["choices"][0]["message"]["content"]
    llm_message = llm_message.replace("```json", "").replace("```", "").strip()

    try:
        response_data = json.loads(llm_message)

        sql_query = response_data.get("sql_query", "").strip()
        explanation = response_data.get("explanation", "")

        if not sql_query:
            raise ValueError("SQL query not found in LLM response.")

        # Validate SQL query to prevent malicious queries
        allowed_statements = ["SELECT", "INSERT", "UPDATE", "DELETE"]
        if not any(sql_query.upper().startswith(stmt) for stmt in allowed_statements):
            raise ValueError("Disallowed SQL statement detected.")

        return {"sql_query": sql_query, "explanation": explanation}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse LLM response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/re_generate/")
def re_generate(request: ReGenerateRequest):
    """Regenerates an improved SQL query and explanation based on user feedback."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=400, detail="Session ID not found. Start a session first."
        )
    schema_response = extract_schema(session_id)
    if "error" in schema_response:
        raise HTTPException(status_code=400, detail=schema_response["error"])

    schemas = schema_response["schemas"]
    schema_texts = [
        f"Table '{schema['table_name']}']:\n{schema['create_statement']}"
        for schema in schemas
    ]
    schema_text = "\n\n".join(schema_texts)
    print("privious question", request.question)
    print("previous query", request.previous_sql_query)
    print("prious explanation", request.previous_explanation)
    print(request.user_description)
    prompt = f"""
    You are an expert SQLite query writer. Here is the database schema:  {schema_text}
    User's Previous Question: {request.question}

    Previous SQL Query: {request.previous_sql_query}
    Previous Explanation: {request.previous_explanation}
    Additional User Description: {request.user_description}

    Based on the additional details provided by the user, regenerate and refine the SQL query for better accuracy and completeness.
    Ensure the query is optimized, syntactically correct, and directly answers the user's question while following these guidelines:

    -- The query must be strictly relevant, including only necessary columns and conditions.
    -- It should follow best practices for performance and readability.
    -- If aggregation, filtering, or ordering is required, include it appropriately.
    -- If assumptions need to be made due to missing details, state them clearly before the query.
    -- Provide only the final query unless additional clarification is necessary.

    After generating the SQL query, analyze and explain it in a detailed yet easy-to-understand manner. Break down the thought process behind constructing the query, including the purpose of each clause, how different components interact, and why specific functions or joins are used.

    Structure the explanation logically with the following sections, and return the entire explanation as a single string with proper line breaks and indentations:

    1. **Query Overview** – Summarize what the query aims to achieve.
    2. **Step-by-Step Breakdown** – Explain each part of the query in sequence, including SELECT, FROM, JOINs, WHERE, GROUP BY, HAVING, ORDER BY, and other clauses.
    3. **Logical Flow** – Describe how data flows through the query, from filtering to aggregation and final output.
    4. **Optimization Considerations** – Discuss performance aspects such as indexing, subqueries vs. joins, or potential improvements.
    5. **Alternative Approaches** – If applicable, suggest different ways to achieve the same result.

    Always use column names in brackets [] without altering the original column names in case of LIKE statements.

    Provide your response in the following JSON format:

    {{
        "re_generated_sql_query": "<YOUR SQL QUERY HERE>",
        "explanation": "<YOUR EXPLANATION HERE>"
    }}
      """

    response = requests.post(
        "https://llmfoundry.straive.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('LLMFOUNDRY_TOKEN')}:my-test-project"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    response_text = response.json()["choices"][0]["message"]["content"]
    response_text = response_text.replace("```json", "").replace("```", "").strip()

    try:
        response_json = json.loads(response_text)
        sql_query = response_json.get("re_generated_sql_query", "").strip()
        explanation = response_json.get("explanation", "").strip()

        return {"re_generated_sql_query": sql_query, "explanation": explanation}
    except (json.JSONDecodeError, KeyError) as e:
        raise HTTPException(status_code=500, detail="Failed to parse LLM response")


@router.post("/run-query/")
def run_query(request: QueryRequest):
    """Executes the generated SQL query for the user's session."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=400, detail="Session ID not found. Start a session first."
        )

    try:
        conn = get_db_connection(session_id)
        df = pd.read_sql_query(request.query, conn)
        result = df.to_dict(orient="records")
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/format-response/")
def format_response(request: FormatResponseRequest):
    """Formats the result into human-readable language using LLM."""
    prompt = f"Format the following raw answer into a well-structured, human-readable response based on the given question. Ensure the response is clear, concise, and appropriately detailed—neither too short nor overly verbose. Preserve all relevant information while improving readability. The response should feel natural and professional.The query from which i got the answer is {request.query} and The raw answer is given below\n\n{request.result}"

    response = requests.post(
        "https://llmfoundry.straive.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('LLMFOUNDRY_TOKEN')}:my-test-project"
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
        },
    )

    human_readable_response = response.json()["choices"][0]["message"]["content"]
    human_readable_response = (
        human_readable_response.replace("```json", "").replace("```", "").strip()
    )
    return {"formatted_answer": human_readable_response}


@router.post("/clear-database/")
def clear_database(request: Request):
    """Clears all tables from the user's session-specific SQLite database."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(
            status_code=400, detail="Session ID not found. Start a session first."
        )

    try:
        conn = get_db_connection(session_id)
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
        tables = cursor.fetchall()

        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table[0]}")

        conn.commit()
        conn.close()
        request.session.clear()
        return {"message": "Database cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=443,
        ssl_keyfile=r"C:\Users\e430275.SPI-GLOBAL\Desktop\anandwork\datachat\private.key",
        ssl_certfile="certificate.pem",
    )

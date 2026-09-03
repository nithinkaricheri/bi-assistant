from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import sqlite3
import requests
import re
import os
import json

app = Flask(__name__)

DB_NAME = "bi_database.db"
UPLOAD_FOLDER = "data"
CSV_PATH = "data/sales.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def clean_columns(df):
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def clean_currency_and_numbers(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            non_nulls = df[col].dropna()
            if non_nulls.empty:
                continue
            
            sample = non_nulls.head(50).astype(str).str.strip()
            
            # Match strings starting with currency symbols ($ currency, rupees, euro, pound) or ending in %
            has_currency = sample.str.contains(r'^[\$₹€£]', regex=True).any()
            has_percent = sample.str.contains(r'%$', regex=True).any()
            has_formatted_number = sample.str.contains(r'^\d{1,3}(,\d{3})+(\.\d+)?$', regex=True).any()
            
            if has_currency or has_percent or has_formatted_number:
                cleaned = df[col].astype(str).str.replace(r'[\$₹€£%,]', '', regex=True).str.strip()
                numeric_series = pd.to_numeric(cleaned, errors='coerce')
                
                # Verify that converting doesn't introduce massive NaN values
                if (numeric_series.isna().sum() - df[col].isna().sum()) < len(df) * 0.2:
                    df[col] = numeric_series
    return df


def save_df_to_db(df, table_name="sales"):
    df = clean_columns(df)
    df = clean_currency_and_numbers(df)
    conn = sqlite3.connect(DB_NAME)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    return df


def load_csv_to_db():
    df = pd.read_csv(CSV_PATH)
    return save_df_to_db(df, "sales")


def init_history_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            sql_query TEXT,
            table_name TEXT,
            status TEXT,
            error_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_query(question, sql, table_name, status, error_msg):
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO query_history (question, sql_query, table_name, status, error_message)
            VALUES (?, ?, ?, ?, ?)
        """, (question, sql, table_name, status, error_msg))
        conn.commit()
        conn.close()
    except Exception as e:
        print("Error logging query:", e)


def get_tables():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'query_history' AND name != 'sqlite_sequence'")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    return tables


def get_table_schema(table_name):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    cols = cur.fetchall()
    conn.close()
    return [{"name": col[1], "type": col[2]} for col in cols]


def get_best_model():
    return "llama3.2:1b"


def ask_ollama(prompt, model):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()["response"]


def clean_sql(text):
    text = text.strip()
    match = re.search(r"```sql(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    else:
        text = text.replace("```", "").strip()
    
    if text.endswith(";"):
        text = text[:-1].strip()
    return text


def validate_sql_safety(sql):
    sql_clean = clean_sql(sql)
    if not sql_clean.lower().startswith("select"):
        return False, "Query must start with SELECT"
    
    blocked = ["insert", "update", "delete", "drop", "alter", "create", "truncate", "pragma", "replace", "vacuum"]
    for block in blocked:
        pattern = r'\b' + re.escape(block) + r'\b'
        if re.search(pattern, sql_clean, re.IGNORECASE):
            return False, f"Forbidden SQL keyword: {block.upper()}"
    return True, None


def check_sql_syntax(sql):
    conn = sqlite3.connect(DB_NAME)
    try:
        cur = conn.cursor()
        cur.execute(f"EXPLAIN {sql}")
        return True, None
    except sqlite3.Error as e:
        return False, str(e)
    finally:
        conn.close()


def generate_sql(question, table_name, model=None):
    schema_str = ""
    cols = get_table_schema(table_name)
    schema_str = "\n".join([f"- {col['name']} ({col['type']})" for col in cols])
    
    conn = sqlite3.connect(DB_NAME)
    try:
        df_sample = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn)
        sample_str = df_sample.to_string(index=False)
    except Exception:
        sample_str = "No sample data available."
    finally:
        conn.close()
    
    prompt = f"""You are an expert SQLite Business Intelligence assistant.
Table Name: {table_name}

Columns and Types:
{schema_str}

Sample Data (first 3 rows):
{sample_str}

User Question:
{question}

IMPORTANT SQLite RULES:
1. Generate ONLY a valid SQLite SELECT query.
2. Return ONLY the raw SQL code. No markdown code blocks (do not wrap in ```sql), no explanation, no other text.
3. Keep the query clean and simple. Use column names exactly as defined above.
4. If performing aggregations (SUM, AVG, COUNT, etc.) along with non-aggregated columns, you MUST group by all the non-aggregated columns.
5. Use descriptive column aliases (e.g., 'total_sales', 'avg_cost').
6. Do not use nested aggregates (e.g. AVG(SUM(col))).
7. To compare text, use LIKE or = correctly.
8. If the question asks for top items or limit, use LIMIT.
"""
    try:
        if not model:
            model = get_best_model()
        result = ask_ollama(prompt, model)
        return clean_sql(result)
    except Exception as e:
        print("Error calling Ollama for SQL generation:", e)
        return None


def run_sql(sql):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def recommend_chart(df):
    if df.empty or df.shape[1] < 2:
        return None
    
    cols = df.columns
    x_col = cols[0]
    y_col = cols[1]
    
    df_clean = df.copy()
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[y_col])
    
    if df_clean.empty:
        return None
        
    df_plot = df_clean.head(50)
    
    is_date = False
    if "date" in str(x_col).lower() or "year" in str(x_col).lower() or "month" in str(x_col).lower():
        is_date = True
        
    num_unique_x = df_plot[x_col].nunique()
    
    chart_type = "bar"
    if is_date:
        chart_type = "line"
    elif num_unique_x <= 7 and df_plot[y_col].min() >= 0:
        chart_type = "pie"
    elif df_plot.shape[0] > 15:
        chart_type = "line"
        
    x_numeric = pd.to_numeric(df_plot[x_col], errors="coerce")
    if not x_numeric.dropna().empty and num_unique_x > 20:
        chart_type = "scatter"
        
    if df_plot.shape[1] >= 3:
        z_col = cols[2]
        z_numeric = pd.to_numeric(df_plot[z_col], errors="coerce")
        if not z_numeric.dropna().empty:
            chart_type = "heatmap"
            
    return chart_type


def format_chart_data(df, chart_type):
    if df.empty or df.shape[1] < 2:
        return None
        
    cols = df.columns
    x_col = cols[0]
    y_col = cols[1]
    
    df_clean = df.copy()
    df_clean[y_col] = pd.to_numeric(df_clean[y_col], errors="coerce")
    df_clean = df_clean.dropna(subset=[y_col])
    
    if df_clean.empty:
        return None
        
    df_plot = df_clean.head(30)
    
    x_data = df_plot[x_col].astype(str).tolist()
    y_data = df_plot[y_col].tolist()
    
    chart_payload = {
        "type": chart_type,
        "x": x_data,
        "y": y_data,
        "x_title": x_col,
        "y_title": y_col,
        "title": f"{y_col} by {x_col}"
    }
    
    if chart_type == "heatmap" and df_plot.shape[1] >= 3:
        try:
            z_col = cols[2]
            df_plot[z_col] = pd.to_numeric(df_plot[z_col], errors="coerce")
            df_pivot = df_plot.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc="mean").fillna(0)
            chart_payload = {
                "type": "heatmap",
                "x": df_pivot.columns.astype(str).tolist(),
                "y": df_pivot.index.astype(str).tolist(),
                "z": df_pivot.values.tolist(),
                "x_title": x_col,
                "y_title": y_col,
                "z_title": z_col,
                "title": f"Heatmap of {z_col} by {x_col} and {y_col}"
            }
        except Exception as e:
            print("Error creating pivot for heatmap:", e)
            
    return chart_payload


def fit_linear_trend(x_indices, y_values):
    n = len(x_indices)
    sum_x = sum(x_indices)
    sum_y = sum(y_values)
    sum_xx = sum(x * x for x in x_indices)
    sum_xy = sum(x * y for x, y in zip(x_indices, y_values))
    
    denom = (n * sum_xx - sum_x * sum_x)
    if denom == 0:
        slope = 0
        intercept = sum_y / n
    else:
        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        
    return slope, intercept


def sanitize_table_name(filename):
    name = os.path.splitext(filename)[0].lower()
    name = re.sub(r'[^a-z0-9_]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if name and name[0].isdigit():
        name = "t_" + name
    if not name:
        name = "uploaded_table"
    return name


def init_db():
    init_history_db()
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sales'")
    exists = cur.fetchone()
    conn.close()
    if not exists:
        try:
            load_csv_to_db()
            print("Default sales dataset initialized.")
        except Exception as e:
            print("Error initializing default dataset:", e)


# --- Flask Routes ---

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/load", methods=["POST"])
def load():
    try:
        df = load_csv_to_db()
        tables = get_tables()
        return jsonify({
            "message": "Default 'sales' dataset loaded successfully",
            "table_name": "sales",
            "columns": df.columns.tolist(),
            "rows": df.head(5).values.tolist(),
            "tables": tables
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})
        
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
        
    filename = file.filename
    sanitized_table = sanitize_table_name(filename)
    
    if not (filename.lower().endswith(".csv") or filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls")):
        return jsonify({"error": "Only CSV and Excel files are allowed"})
        
    try:
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(upload_path)
        else:
            df = pd.read_excel(upload_path)
            
        df = save_df_to_db(df, sanitized_table)
        tables = get_tables()
        
        return jsonify({
            "message": f"Successfully loaded table '{sanitized_table}'",
            "table_name": sanitized_table,
            "columns": df.columns.tolist(),
            "rows": df.head(5).values.tolist(),
            "tables": tables
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/models", methods=["GET"])
def get_models_route():
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if res.status_code == 200:
            models_info = res.json()
            models = [m["name"] for m in models_info.get("models", [])]
            if not models:
                models = ["llama3.2:1b", "llama3.2:latest", "llama3:latest"]
            return jsonify({"models": models})
        return jsonify({"models": ["llama3.2:1b", "llama3.2:latest", "llama3:latest"]})
    except Exception:
        return jsonify({"models": ["llama3.2:1b", "llama3.2:latest", "llama3:latest"]})


@app.route("/tables", methods=["GET"])
def get_tables_route():
    try:
        return jsonify({"tables": get_tables()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/table_schema/<table_name>", methods=["GET"])
def table_schema(table_name):
    try:
        tables = get_tables()
        if table_name not in tables:
            return jsonify({"error": f"Table '{table_name}' not found"}), 404
            
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(f'PRAGMA table_info("{table_name}")')
        cols = [{"name": c[1], "type": c[2]} for c in cur.fetchall()]
        
        cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        row_count = cur.fetchone()[0]
        
        df_sample = pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT 100', conn)
        conn.close()
        
        return jsonify({
            "table_name": table_name,
            "columns": cols,
            "row_count": row_count,
            "sample_rows": df_sample.values.tolist(),
            "sample_columns": df_sample.columns.tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/table_kpi/<table_name>", methods=["GET"])
def table_kpi(table_name):
    try:
        tables = get_tables()
        if table_name not in tables:
            return jsonify({"error": f"Table '{table_name}' not found"}), 404
            
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(f'PRAGMA table_info("{table_name}")')
        cols = cur.fetchall()
        
        numeric_cols = []
        for c in cols:
            col_name = c[1]
            col_type = c[2].upper()
            if "INT" in col_type or "REAL" in col_type or "NUM" in col_type or "FLOAT" in col_type or "DOUBLE" in col_type:
                numeric_cols.append(col_name)
                
        cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        total_rows = cur.fetchone()[0]
        
        kpis = []
        
        # Smart Dynamic Column Matching
        revenue_col = next((c for c in numeric_cols if c.lower() in ["sales", "revenue", "amount", "total"]), None)
        if not revenue_col:
            revenue_col = next((c for c in numeric_cols if any(w in c.lower() for w in ["sales", "revenue", "amount", "total", "price"])), None)
            
        qty_col = next((c for c in numeric_cols if c.lower() in ["quantity", "qty", "units", "items"]), None)
        if not qty_col:
            qty_col = next((c for c in numeric_cols if any(w in c.lower() for w in ["quantity", "qty", "units", "items", "count", "sold"])), None)
            
        cost_col = next((c for c in numeric_cols if c.lower() in ["cost", "expenses", "costs"]), None)
        if not cost_col:
            cost_col = next((c for c in numeric_cols if any(w in c.lower() for w in ["cost", "expense"])), None)
            
        discount_col = next((c for c in numeric_cols if c.lower() in ["discount", "disc"]), None)
        if not discount_col:
            discount_col = next((c for c in numeric_cols if any(w in c.lower() for w in ["discount", "disc", "percentage", "pct"])), None)
            
        profit_col = next((c for c in numeric_cols if c.lower() in ["profit", "gain", "margin"]), None)
        if not profit_col:
            profit_col = next((c for c in numeric_cols if any(w in c.lower() for w in ["profit", "gain", "margin"])), None)
        
        if revenue_col:
            cur.execute(f"SELECT SUM(\"{revenue_col}\") FROM \"{table_name}\"")
            total_rev = cur.fetchone()[0] or 0
            kpis.append({"label": f"Total {revenue_col.replace('_', ' ').title()}", "value": f"${total_rev:,.2f}" if total_rev > 100 else f"{total_rev:,}", "icon": "dollar-sign"})
            
        if profit_col:
            cur.execute(f"SELECT SUM(\"{profit_col}\") FROM \"{table_name}\"")
            total_prof = cur.fetchone()[0] or 0
            kpis.append({"label": f"Total {profit_col.replace('_', ' ').title()}", "value": f"${total_prof:,.2f}" if total_prof > 100 else f"{total_prof:,}", "icon": "trending-up"})
            
        if qty_col:
            cur.execute(f"SELECT SUM(\"{qty_col}\") FROM \"{table_name}\"")
            total_qty = cur.fetchone()[0] or 0
            kpis.append({"label": f"Total {qty_col.replace('_', ' ').title()}", "value": f"{total_qty:,}", "icon": "shopping-cart"})
            
        if discount_col:
            cur.execute(f"SELECT AVG(\"{discount_col}\") FROM \"{table_name}\"")
            avg_disc = cur.fetchone()[0] or 0
            if avg_disc < 1.0 and avg_disc > 0:
                kpis.append({"label": f"Avg {discount_col.replace('_', ' ').title()}", "value": f"{avg_disc * 100:.1f}%", "icon": "tag"})
            else:
                kpis.append({"label": f"Avg {discount_col.replace('_', ' ').title()}", "value": f"{avg_disc:.1f}%", "icon": "tag"})
                
        for col in numeric_cols:
            if len(kpis) >= 4:
                break
            if col in [revenue_col, qty_col, cost_col, discount_col, profit_col]:
                continue
            cur.execute(f"SELECT AVG(\"{col}\") FROM \"{table_name}\"")
            val = cur.fetchone()[0] or 0
            kpis.append({"label": f"Avg {col.replace('_', ' ').title()}", "value": f"{val:,.1f}", "icon": "activity"})
            
        conn.close()
        
        if len(kpis) < 3:
            kpis.append({"label": "Total Columns", "value": str(len(cols)), "icon": "table"})
            
        return jsonify({
            "table_name": table_name,
            "total_rows": total_rows,
            "column_count": len(cols),
            "kpis": kpis
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete_table/<table_name>", methods=["DELETE"])
def delete_table(table_name):
    try:
        if table_name in ["query_history", "sqlite_sequence"]:
            return jsonify({"error": "Cannot delete system tables"}), 400
            
        tables = get_tables()
        if table_name not in tables:
            return jsonify({"error": f"Table '{table_name}' not found"}), 404
            
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute(f'DROP TABLE "{table_name}"')
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": f"Table '{table_name}' deleted successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.json or {}
        question = data.get("question")
        table_name = data.get("table_name", "sales")
        model = data.get("model")
        
        if not question:
            return jsonify({"error": "Question is required"}), 400
            
        tables = get_tables()
        if not tables:
            return jsonify({"error": "No datasets available. Please load a dataset first."}), 400
        if table_name not in tables:
            table_name = tables[0]
            
        sql = generate_sql(question, table_name, model=model)
        
        if sql is None:
            log_query(question, "NULL", table_name, "error", "AI failed to generate SQL.")
            return jsonify({"error": "AI could not generate a SQL query. Please rephrase."}), 400
            
        is_safe, safety_err = validate_sql_safety(sql)
        if not is_safe:
            log_query(question, sql, table_name, "error", f"Safety block: {safety_err}")
            return jsonify({"error": f"Generated SQL query was blocked for security: {safety_err}", "sql": sql}), 400
            
        is_valid_syntax, syntax_err = check_sql_syntax(sql)
        if not is_valid_syntax:
            log_query(question, sql, table_name, "error", f"Syntax error: {syntax_err}")
            return jsonify({"error": f"Generated SQL contains a syntax error: {syntax_err}", "sql": sql}), 400
            
        df = run_sql(sql)
        chart_type = recommend_chart(df)
        chart = format_chart_data(df, chart_type)
        
        log_query(question, sql, table_name, "success", None)
        
        return jsonify({
            "sql": sql,
            "table_name": table_name,
            "columns": df.columns.tolist(),
            "rows": df.values.tolist(),
            "chart": chart,
            "chart_type": chart_type
        })
    except Exception as e:
        try:
            log_query(question if 'question' in locals() else "Unknown Question", 
                      sql if 'sql' in locals() else "NULL", 
                      table_name if 'table_name' in locals() else "sales", 
                      "error", str(e))
        except Exception:
            pass
        return jsonify({"error": str(e), "sql": sql if 'sql' in locals() else None}), 500


@app.route("/run_custom_sql", methods=["POST"])
def run_custom_sql():
    try:
        data = request.json or {}
        sql = data.get("sql")
        question = data.get("question", "Custom SQL Run")
        table_name = data.get("table_name", "sales")
        
        if not sql:
            return jsonify({"error": "SQL query is required"}), 400
            
        sql = clean_sql(sql)
        is_safe, safety_err = validate_sql_safety(sql)
        if not is_safe:
            return jsonify({"error": f"Security validation failed: {safety_err}"}), 400
            
        is_syntax_valid, syntax_err = check_sql_syntax(sql)
        if not is_syntax_valid:
            return jsonify({"error": f"SQL syntax error: {syntax_err}"}), 400
            
        df = run_sql(sql)
        chart_type = recommend_chart(df)
        chart = format_chart_data(df, chart_type)
        
        log_query(question, sql, table_name, "success", None)
        
        return jsonify({
            "sql": sql,
            "columns": df.columns.tolist(),
            "rows": df.values.tolist(),
            "chart": chart,
            "chart_type": chart_type
        })
    except Exception as e:
        try:
            log_query(question if 'question' in locals() else "Custom SQL Run", 
                      sql if 'sql' in locals() else "NULL", 
                      table_name if 'table_name' in locals() else "sales", 
                      "error", str(e))
        except Exception:
            pass
        return jsonify({"error": str(e), "sql": sql if 'sql' in locals() else None}), 500


@app.route("/insight_stream", methods=["POST"])
def insight_stream():
    try:
        data = request.json or {}
        question = data.get("question")
        table_name = data.get("table_name", "sales")
        rows = data.get("rows", [])
        columns = data.get("columns", [])
        model = data.get("model") or get_best_model()
        
        df = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame()
        df_summary = df.head(15).to_string(index=False)
        
        prompt = f"""You are an elite business intelligence analyst and strategist.
Analyzed Table: {table_name}
User Question: {question}

Query Results (Top 15 rows):
{df_summary}

Based on these results:
1. Summarize the key findings in 2-3 concise, impactful bullet points.
2. Provide 2 actionable strategic business recommendations.
3. Highlight any anomalies, interesting trends, or metrics (e.g. highest performing, lowest performing).

Format your response in clean Markdown. Be direct, professional, and insight-driven. Do not repeat the table itself."""
        
        def generate():
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True
            }
            try:
                response = requests.post(url, json=payload, stream=True, timeout=180)
                for line in response.iter_lines():
                    if line:
                        json_data = json.loads(line.decode('utf-8'))
                        chunk = json_data.get("response", "")
                        yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                        if json_data.get("done", False):
                            break
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                
        return Response(generate(), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forecast", methods=["POST"])
def forecast():
    try:
        data = request.json or {}
        rows = data.get("rows", [])
        columns = data.get("columns", [])
        
        if not rows or len(columns) < 2:
            return jsonify({"error": "Insufficient data for forecasting. Please run a query with at least a time/category column and a numeric value column."}), 400
            
        df = pd.DataFrame(rows, columns=columns)
        x_col = columns[0]
        y_col = columns[1]
        
        df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
        df = df.dropna(subset=[x_col, y_col])
        
        if df.shape[0] < 3:
            return jsonify({"error": "Need at least 3 historical points to project a trend."}), 400
            
        is_date = False
        try:
            df['parsed_date'] = pd.to_datetime(df[x_col], errors='coerce')
            if not df['parsed_date'].dropna().empty:
                df = df.dropna(subset=['parsed_date'])
                df = df.sort_values(by='parsed_date')
                is_date = True
        except Exception:
            pass
            
        n = df.shape[0]
        indices = list(range(n))
        y_vals = df[y_col].values.tolist()
        
        slope, intercept = fit_linear_trend(indices, y_vals)
        
        project_count = min(10, max(3, int(n * 0.3)))
        future_indices = list(range(n, n + project_count))
        future_y = [slope * idx + intercept for idx in future_indices]
        
        min_y = min(y_vals)
        if min_y >= 0:
            future_y = [max(0, val) for val in future_y]
            
        if is_date:
            last_date = df['parsed_date'].iloc[-1]
            diffs = df['parsed_date'].diff().dropna()
            freq_days = diffs.mean().days if not diffs.empty else 1
            if freq_days <= 0:
                freq_days = 1
            future_dates = [last_date + pd.Timedelta(days=int(freq_days * (i + 1))) for i in range(project_count)]
            future_x = [d.strftime('%Y-%m-%d') for d in future_dates]
        else:
            future_x = [f"Proj {i+1}" for i in range(project_count)]
            
        history_x = df[x_col].astype(str).tolist()
        history_y = y_vals
        
        forecast_x = [history_x[-1]] + list(future_x)
        forecast_y = [history_y[-1]] + list(future_y)
        
        return jsonify({
            "history_x": history_x,
            "history_y": history_y,
            "forecast_x": forecast_x,
            "forecast_y": forecast_y,
            "x_title": x_col,
            "y_title": y_col,
            "title": f"Forecast for {y_col}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history_endpoint():
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("SELECT id, question, sql_query, table_name, status, error_message, timestamp FROM query_history ORDER BY id DESC LIMIT 50")
        rows = cur.fetchall()
        conn.close()
        
        history = []
        for r in rows:
            history.append({
                "id": r[0],
                "question": r[1],
                "sql_query": r[2],
                "table_name": r[3],
                "status": r[4],
                "error_message": r[5],
                "timestamp": r[6]
            })
            
        return jsonify({"history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    init_db()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
    )
# raw_data.py
import pandas as pd
import re
import gdown
from sqlalchemy import create_engine, text

# Step 0: Download files from Google Drive
def download_from_gdrive():
    csv_url = "https://drive.google.com/uc?id=1paFY1Fz53NfjQF64WHySC3-AMQ0ogJyy"  # 2015.csv
    rules_url = "https://drive.google.com/uc?id=1TP-ezidA1iF2saBzwqTfTVs-ByjBkRtH"  # Book 5.csv

    print("Downloading 2015.csv from Google Drive...")
    gdown.download(csv_url, "2015.csv", quiet=False)

    print("Downloading Book 5.csv from Google Drive...")
    gdown.download(rules_url, "Book 5.csv", quiet=False)

# Load CSV in chunks
def load_csv_to_db_in_batches(file_path, chunksize, table_name, engine, usecols=None):
    total_rows = 0
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize, usecols=usecols)):
        chunk.columns = [col.lower() for col in chunk.columns]
        chunk.to_sql(table_name, engine, if_exists='append' if i > 0 else 'replace', index=False)
        total_rows += len(chunk)
        print(f"Chunk {i+1}: Loaded {len(chunk)} rows")
    print(f"Total rows uploaded: {total_rows}")

# Rule parsing
def parse_rule_to_sql_condition(column, rule_text):
    if not isinstance(rule_text, str):
        return None

    rule_text = rule_text.replace('–', '-')  # normalize dash
    range_matches = re.findall(r"(\d+)-(\d+)", rule_text)
    allowed_values = set()

    for start, end in range_matches:
        allowed_values.update(range(int(start), int(end) + 1))

    numbers = set(map(int, re.findall(r"\b\d+\b", rule_text)))
    allowed_values.update(numbers)

    if not allowed_values:
        return None

    allowed_values = sorted(allowed_values)
    values_str = ', '.join(str(v) for v in allowed_values)
    column_lower = column.lower()

    return f"({column_lower}) NOT IN ({values_str})"

def main():
    try:
        # Step 0: Download files
        download_from_gdrive()

        # ------------------ Config -------------------
        csv_file_path = "2015.csv"
        rules_excel_path = "Book 5.csv"
        chunksize = 20000
        table_name = "raw_table"

        # ---------- Supabase PostgreSQL Connection ----------
        SUPABASE_DB_URL = "postgresql://postgres.atlvdqqnccdrenirkskg:saira123@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
        engine = create_engine(SUPABASE_DB_URL)

        # ---------- Columns to load from CSV ----------
        columns_to_load = [
            'SEX', '_AGEG5YR', 'EDUCA', 'INCOME2', 'EMPLOY1', 'MARITAL', 'RENTHOM1',
            'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2',
            'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'BPMEDS', 'CHOLCHK', 'TOLDHI2', 'DIABETE3'
        ]

        # ---------- Step 1: Upload CSV ----------
        print("Uploading CSV to Supabase in chunks...")
        load_csv_to_db_in_batches(csv_file_path, chunksize, table_name, engine, usecols=columns_to_load)

        # ---------- Step 2: Load Validation Rules ----------
        rules_df = pd.read_excel(rules_excel_path)
        rules_df.columns = (
            rules_df.columns
            .str.strip()
            .str.replace('\n', '', regex=True)
            .str.lower()
            .str.replace(' ', '_')
        )

        if 'source_column_name' not in rules_df.columns:
            possible_cols = [col for col in rules_df.columns if col.startswith('source_column_name')]
            if possible_cols:
                rules_df.rename(columns={possible_cols[0]: 'source_column_name'}, inplace=True)
            else:
                raise KeyError("Column 'source_column_name' not found and no similar column found.")

        rules_df = rules_df.head(20)
        print("Validation rules loaded.")

        # ---------- Step 3: Apply Rules ----------
        all_invalid_rows = pd.DataFrame()

        for idx, row in rules_df.iterrows():
            col = row['source_column_name']
            rule = row['validation_rule']

            if pd.isna(col) or pd.isna(rule):
                continue

            condition = parse_rule_to_sql_condition(col, rule)
            if not condition:
                print(f"Skipping rule for '{col}': could not parse '{rule}'")
                continue

            sql_query = f"SELECT * FROM {table_name} WHERE {condition}"
            delete_query = f"DELETE FROM {table_name} WHERE {condition}"
            print(f"Validating column '{col}' with rule: {rule}")

            try:
                invalid_rows = pd.read_sql_query(sql_query, engine)
            except Exception as e:
                print(f"Error querying column '{col}': {e}")
                continue

            if invalid_rows.empty:
                print(f"All rows valid for '{col}'.")
            else:
                print(f"Found {len(invalid_rows)} invalid rows for '{col}'.")
                invalid_rows['invalid_column'] = col
                invalid_rows['violated_rule'] = rule
                all_invalid_rows = pd.concat([all_invalid_rows, invalid_rows], ignore_index=True)
                print(f"Deleting invalid rows for '{col}'...")
                with engine.begin() as conn:
                    conn.execute(text(delete_query))

        # Step 4: Save Invalid Rows
        if not all_invalid_rows.empty:
            all_invalid_rows.drop_duplicates().to_csv("invalid_rows1.csv", index=False)
            print("Invalid rows saved to 'invalid_rows1.csv'")

    except Exception as e:
        print(f"[ERROR]: {str(e)}")

if __name__ == "__main__":
    main()

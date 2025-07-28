import pandas as pd
from sqlalchemy import create_engine, types, text
from cryptography.fernet import Fernet
import numpy as np
import re
import gdown  # To fetch Excel from Google Drive

# --- Download Excel Rules from Google Drive ---
gdown.download("https://drive.google.com/uc?id=1TP-ezidA1iF2saBzwqTfTVs-ByjBkRtH", "Book 5.xlsx", quiet=False)

# --- Load rules and setup cipher ---
rules_df = pd.read_excel('Book 5.xlsx')
rules_df.fillna('', inplace=True)
rules = rules_df.set_index('presentation_layer_column_name').T.to_dict()

key = Fernet.generate_key()
cipher = Fernet(key)
print(f"Fernet key (save this to decrypt later): {key.decode()}")

# --- Helper: batch upload ---
def upload_df_to_db_in_batches(df, chunksize, table_name, engine, dtype=None):
    total_rows = 0
    num_chunks = (len(df) + chunksize - 1) // chunksize

    for i in range(num_chunks):
        chunk = df.iloc[i * chunksize:(i + 1) * chunksize]
        chunk.to_sql(
            table_name,
            engine,
            if_exists='append' if i > 0 else 'replace',
            index=False,
            dtype=dtype
        )
        total_rows += len(chunk)
        print(f"Chunk {i + 1}/{num_chunks}: Loaded {len(chunk)} rows")

    print(f"Total rows uploaded: {total_rows}")

# --- Helper: parse validation rules to SQL condition ---
def parse_rule_to_sql_condition(column, rule_text):
    if not isinstance(rule_text, str):
        return None

    rule_text = rule_text.replace(',', '')  # normalize
    range_matches = re.findall(r'(\d+)\s*-\s*(\d+)', rule_text)
    allowed_values = set()

    for start, end in range_matches:
        allowed_values.update(range(int(start), int(end) + 1))

    numbers = set(map(int, re.findall(r'\b\d+\b', rule_text)))
    allowed_values.update(numbers)

    if not allowed_values:
        return None

    values_str = ', '.join(str(v) for v in sorted(allowed_values))
    return f'"{column}" NOT IN ({values_str})'

# --- Supabase DB connection ---
SUPABASE_DB_URL = "postgresql://postgres.atlvdqqnccdrenirkskg:saira123@aws-0-ap-southeast-1.pooler.supabase.com:5432/postgres"
presentation_engine = create_engine(SUPABASE_DB_URL)
staging_engine = create_engine(SUPABASE_DB_URL)

# --- Load staging data ---
query = "SELECT * FROM staging_table;"
df = pd.read_sql_query(query, staging_engine)
print("Loaded data from staging_table, shape:", df.shape)

# --- Rename columns ---
rename_map = {
    "Gender": "gender",
    "AgeGroup": "agegroup",
    "EducationLevel": "educationlevel",
    "IncomeLevel": "incomelevel",
    "EmploymentStatus": "employmentstatus",
    "Marital status": "maritalstatus",
    "HomeOwnership": "homeownership",
    "GeneralHealth": "generalhealth",
    "PhysicalHealth DaysBad": "physicalhealth_daysbad",
    "MentalHealth DaysBad": "mentalhealth_daysbad",
    "HealthLimitedDays": "healthlimiteddays",
    "HasHealthPlan": "hashealthplan",
    "HasPersonalDoctor": "haspersonaldoctor",
    "CouldNotAffordDoctor": "couldnotafforddoctor",
    "LastRoutineCheckup": "lastroutinecheckup",
    "HighBloodPressure": "highbloodpressure",
    "OnBloodPressureMeds": "onbloodpressuremeds",
    "CholesterolChecked": "cholesterolchecked",
    "HighCholesterol": "highcholesterol",
    "Diabetes": "diabetes",
}
df.rename(columns=rename_map, inplace=True)

# --- Apply privacy & validation rules ---
def apply_rules(df, rules):
    processed_df = pd.DataFrame(index=df.index)

    for col in df.columns:
        rule = rules.get(col, {})
        show = rule.get('show_in_presentation', '').strip().lower() in ['true', 'yes']
        if show or col not in rules:
            processed_df[col] = df[col]

    for col, rule in rules.items():
        if col not in df.columns:
            continue

        show = rule.get('show_in_presentation', '').strip().lower() in ['true', 'yes']
        if not show:
            processed_df.drop(columns=[col], errors='ignore', inplace=True)
            continue

        new_col = rule.get("new_column_name", '') or col
        merge_with = rule.get('merge_with_column', '')
        action = rule.get('privacy_action', '').lower()
        validation = rule.get('validation_rule', '').lower()

        if merge_with and merge_with in df.columns:
            processed_df[new_col] = df[col].astype(str) + df[merge_with].astype(str)
            processed_df.drop(columns=[col], errors='ignore', inplace=True)
            continue

        if action == 'mask':
            processed_df[new_col] = df[col].astype(str).str[0] + '*****'
        elif action == 'encrypt':
            processed_df[new_col] = df[col].apply(lambda x: cipher.encrypt(str(x).encode()).decode() if pd.notna(x) else x)
        else:
            processed_df[new_col] = df[col]

        if validation == 'not_null':
            processed_df[new_col] = processed_df[new_col].fillna('MISSING')
        elif validation == 'valid_email_regex':
            processed_df[new_col] = processed_df[new_col].where(processed_df[new_col].str.contains(r'^\S+@\S+\.\S+$', na=False), np.nan)
        elif validation.startswith('min_'):
            min_val = int(validation.split('_')[1])
            processed_df[new_col] = processed_df[new_col].where(processed_df[new_col] >= min_val)
        elif validation.startswith('greater_than_'):
            gt_val = int(validation.split('_')[-1])
            processed_df[new_col] = processed_df[new_col].where(processed_df[new_col] > gt_val)

    return processed_df

df = apply_rules(df, rules)
df.insert(0, 'id', range(1, len(df) + 1))

# --- Define SQL types ---
sql_types = {}
from sqlalchemy import INTEGER, SMALLINT, TEXT

for col in df.columns:
    if col == 'id':
        sql_types[col] = INTEGER()
    else:
        try:
            col_numeric = pd.to_numeric(df[col], errors='coerce')
            if col_numeric.notna().any():
                max_val = col_numeric.max()
                sql_types[col] = SMALLINT() if max_val <= 32767 else INTEGER()
            else:
                sql_types[col] = TEXT()
        except:
            sql_types[col] = TEXT()

# --- Upload to presentation table ---
upload_df_to_db_in_batches(df, 20000, 'presentation', presentation_engine, dtype=sql_types)

# --- Validate again & delete invalid ---
all_invalid_rows = pd.DataFrame()

for _, rule_row in rules_df.iterrows():
    col = rule_row.get('presentation_layer_column_name')
    rule_text = rule_row.get('validation_rule')
    action = rule_row.get('privacy_action', '').lower()

    if not col or not rule_text or col not in df.columns or action in ['encrypt', 'mask']:
        continue

    condition = parse_rule_to_sql_condition(col, rule_text)
    if not condition:
        continue

    sql_query = f"SELECT * FROM presentation WHERE {condition};"
    delete_query = f"DELETE FROM presentation WHERE {condition};"

    invalid = pd.read_sql_query(sql_query, presentation_engine)
    if not invalid.empty:
        invalid['invalid_column'] = col
        invalid['violated_rule'] = rule_text
        all_invalid_rows = pd.concat([all_invalid_rows, invalid], ignore_index=True)
        with presentation_engine.begin() as conn:
            conn.execute(text(delete_query))
        print(f"Deleted {len(invalid)} invalid rows from presentation.")

# --- Save invalid rows ---
if not all_invalid_rows.empty:
    all_invalid_rows.drop_duplicates(inplace=True)
    all_invalid_rows.to_csv("invalid_presentation_rows.csv", index=False)
    print("Saved invalid rows to 'invalid_presentation_rows.csv'")
else:
    print("No invalid rows to remove.")

# --- Add Primary Key ---
with presentation_engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE presentation ADD PRIMARY KEY (id);"))
    except Exception as e:
        print(f"Primary key may already exist: {e}")

print("Presentation layer processing complete!")

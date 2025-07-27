import pandas as pd
from sqlalchemy import create_engine, types, text
from cryptography.fernet import Fernet
import numpy as np
import re

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

# --- Database connection parameters ---
username = 'postgres'
password = 'postgre'
host = 'localhost'
port = '5432'

# --- Connect to databases ---
staging_engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/staging_data')
presentation_engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/presentation_data')

# --- Load data from staging table ---
query = "SELECT * FROM staging_table;"
df = pd.read_sql_query(query, staging_engine)
print("Loaded data from staging_data.staging_table, shape:", df.shape)

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
print("Columns after renaming:", df.columns.tolist())

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
            if col in processed_df.columns:
                processed_df.drop(columns=[col], inplace=True)
            continue

        new_col = rule.get("new_column_name", '') or col
        merge_with = rule.get('merge_with_column', '')
        action = rule.get('privacy_action', '').lower()
        validation = rule.get('validation_rule', '').lower()

        if merge_with and merge_with in df.columns:
            merged = df[col].astype(str) + df[merge_with].astype(str)
            processed_df[new_col] = merged
            if new_col != col and col in processed_df.columns:
                processed_df.drop(columns=[col], inplace=True)
            continue

        if action == 'mask':
            processed_df[new_col] = df[col].astype(str).str[0] + '*****'
        elif action == 'encrypt':
            processed_df[new_col] = df[col].apply(
                lambda x: cipher.encrypt(str(x).encode()).decode() if pd.notna(x) else x
            )
        else:
            processed_df[new_col] = df[col]

        if validation == 'not_null':
            processed_df[new_col] = processed_df[new_col].fillna('MISSING')
        elif validation == 'valid_email_regex':
            processed_df[new_col] = processed_df[new_col].where(
                processed_df[new_col].str.contains(r'^\S+@\S+\.\S+$', na=False), np.nan
            )
        elif validation.startswith('min_'):
            min_val = int(validation.split('_')[1])
            processed_df[new_col] = processed_df[new_col].where(processed_df[new_col] >= min_val)
        elif validation.startswith('greater_than_'):
            gt_val = int(validation.split('_')[-1])
            processed_df[new_col] = processed_df[new_col].where(processed_df[new_col] > gt_val)

        if col == 'statecode':
            numeric_col = pd.to_numeric(processed_df[new_col], errors='coerce')
            processed_df.loc[numeric_col.between(1, 72), new_col] = np.nan

        if new_col != col and col in processed_df.columns:
            processed_df.drop(columns=[col], inplace=True)

    for col, rule in rules.items():
        new_col = rule.get('new_column_name', '') or col
        action = rule.get('privacy_action', '').lower()
        show = rule.get('show_in_presentation', '').strip().lower() in ['true', 'yes']

        if show and action in ['encrypt', 'mask'] and new_col in processed_df.columns:
            processed_df[new_col] = processed_df[new_col].astype(str)

    return processed_df

# Apply the rules
df = apply_rules(df, rules)
print("Columns after apply_rules:", df.columns.tolist())

# Add ID column
df.insert(0, 'id', range(1, len(df) + 1))

# --- Build SQL types ---
from sqlalchemy import INTEGER, SMALLINT, TEXT

sql_types = {}
SMALLINT_MIN = -32768
SMALLINT_MAX = 32767

for col in df.columns:
    col_type = TEXT()
    if col == 'id':
        col_type = INTEGER()
    else:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        if numeric_col.notna().any():
            min_val = numeric_col.min()
            max_val = numeric_col.max()
            if (min_val > SMALLINT_MIN) and (max_val <= SMALLINT_MAX):
                col_type = SMALLINT()
            else:
                col_type = INTEGER()
    sql_types[col] = col_type

# Override types for encrypted/masked columns
for col, rule in rules.items():
    new_col = rule.get('new_column_name', '') or col
    action = rule.get('privacy_action', '').lower()
    show = rule.get('show_in_presentation', '').strip().lower() in ['true', 'yes']
    if show and action in ['encrypt', 'mask']:
        sql_types[new_col] = TEXT()

# --- Upload to presentation table ---
chunksize = 20000
print("Uploading full data to presentation table in batches...")
upload_df_to_db_in_batches(df, chunksize, 'presentation', presentation_engine, dtype=sql_types)

# --- Validate and clean invalid rows ---
all_invalid_rows = pd.DataFrame()

for idx, rule_row in rules_df.iterrows():
    col = rule_row.get('presentation_layer_column_name')
    validation_rule = rule_row.get('validation_rule')
    privacy_action = rule_row.get('privacy_action', '').lower()

    if not col or not validation_rule or col not in df.columns:
        continue

    if privacy_action in ['encrypt', 'mask']:
        print(f"Skipping validation for '{col}' because it is {privacy_action}.")
        continue

    condition = parse_rule_to_sql_condition(col, validation_rule)
    if not condition:
        print(f"Skipping validation for '{col}': cannot parse rule '{validation_rule}'")
        continue

    sql_query = f"SELECT * FROM presentation WHERE {condition};"
    delete_query = f"DELETE FROM presentation WHERE {condition};"

    print(f"Validating column '{col}' with rule: '{validation_rule}'")
    invalid_rows = pd.read_sql_query(sql_query, presentation_engine)

    if invalid_rows.empty:
        print(f"All rows valid for '{col}'.")
    else:
        print(f"Found {len(invalid_rows)} invalid rows for '{col}'.")
        invalid_rows['invalid_column'] = col
        invalid_rows['violated_rule'] = validation_rule
        all_invalid_rows = pd.concat([all_invalid_rows, invalid_rows], ignore_index=True)

        with presentation_engine.begin() as conn:
            conn.execute(text(delete_query))
            print(f"Deleted {len(invalid_rows)} invalid rows from presentation table.")

# --- Save invalid rows ---
if not all_invalid_rows.empty:
    all_invalid_rows.drop_duplicates(inplace=True)
    all_invalid_rows.to_csv("invalid_presentation_rows.csv", index=False)
    print("Invalid rows saved to 'invalid_presentation_rows.csv'")
else:
    print("No invalid rows found or deleted.")

# --- Create primary key ---
with presentation_engine.connect() as connection:
    try:
        connection.execute(text('ALTER TABLE presentation ADD PRIMARY KEY (id);'))
    except Exception as e:
        print(f"Warning: could not add primary key (maybe it already exists); {e}")

    result = connection.execute(text(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'presentation';"
    ))
    existing_columns = [row[0] for row in result]
    print("Presentation data processing, validation, batch loading complete!")

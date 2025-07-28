import pandas as pd
import os
import re
import smtplib
import gdown
from sqlalchemy import create_engine, text, types

# Download files from Google Drive
def download_file_from_drive(drive_url, output_name):
    file_id = drive_url.split("/d/")[1].split("/")[0]
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_name, quiet=False)

# Rule Parsing Function
def parse_rule_to_sql_condition(column, rule_text):
    if not isinstance(rule_text, str):
        return None
    rule_text = rule_text.replace('–', '-')
    range_matches = re.findall(r"(\d+)-(\d+)", rule_text)
    allowed_values = set()
    for start, end in range_matches:
        allowed_values.update(range(int(start), int(end) + 1))
    numbers = set(map(int, re.findall(r"\b\d+\b", rule_text)))
    allowed_values.update(numbers)
    if not allowed_values:
        return None
    values_str = ', '.join(str(v) for v in sorted(allowed_values))
    return f"({column}) NOT IN ({values_str})"

# Upload in batches
def upload_df_to_db_in_batches(df, chunksize, table_name, engine, dtype=None):
    total_rows = 0
    for i in range(0, len(df), chunksize):
        chunk = df.iloc[i:i+chunksize]
        chunk.to_sql(table_name, engine, if_exists='append' if i > 0 else 'replace', index=False, dtype=dtype)
        total_rows += len(chunk)
        print(f"Chunk {i//chunksize+1}: Uploaded {len(chunk)} rows")
    print(f"Total rows uploaded: {total_rows}")

def main():
    try:
        # --- Supabase connection (STAGING DB) ---
        username = 'postgres.atlvdqqnccdrenirkskg'
        password = 'saira123'
        host = 'aws-0-ap-southeast-1.pooler.supabase.com'
        port = '5432'
        database_staging = 'postgres'

        # --- Create Engine ---
        staging_engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database_staging}')

        # --- Download Required Files ---
        download_file_from_drive("https://drive.google.com/file/d/1paFY1Fz53NfjQF64WHySC3-AMQ0ogJyy/view?usp=sharing", "2015.csv")
        download_file_from_drive("https://drive.google.com/file/d/1TP-ezidA1iF2saBzwqTfTVs-ByjBkRtH/view?usp=sharing", "Book 5.xlsx")

        # --- Load Raw Data ---
        raw_df = pd.read_csv("2015.csv", usecols=[
            'SEX', '_AGEG5YR', 'EDUCA', 'INCOME2', 'EMPLOY1', 'MARITAL', 'RENTHOM1',
            'GENHLTH', 'PHYSHLTH', 'MENTHLTH', 'POORHLTH', 'HLTHPLN1', 'PERSDOC2',
            'MEDCOST', 'CHECKUP1', 'BPHIGH4', 'BPMEDS', 'CHOLCHK', 'TOLDHI2', 'DIABETE3'
        ])
        raw_df.columns = [col.lower() for col in raw_df.columns]

        # --- Rename Columns ---
        rename_map = {
            'sex': 'Gender',
            '_ageg5yr': 'AgeGroup',
            'educa': 'EducationLevel',
            'income2': 'IncomeLevel',
            'employ1': 'EmploymentStatus',
            'marital': 'MaritalStatus',
            'renthom1': 'HomeOwnership',
            'genhlth': 'GeneralHealth',
            'physhlth': 'PhysicalHealth_DaysBad',
            'menthlth': 'MentalHealth_DaysBad',
            'poorhlth': 'HealthLimitedDays',
            'hlthpln1': 'HasHealthPlan',
            'persdoc2': 'HasPersonalDoctor',
            'medcost': 'CouldNotAffordDoctor',
            'checkup1': 'LastRoutineCheckup',
            'bphigh4': 'HighBloodPressure',
            'bpmeds': 'MedForBP',
            'cholchk': 'CholesterolChecked',
            'toldhi2': 'HighCholesterol',
            'diabete3': 'Diabetes'
        }
        df = raw_df.rename(columns=rename_map)

        df.dropna(how='all', axis=1, inplace=True)
        df = df.where(pd.notnull(df), None)

        # --- Define SQL Types ---
        sql_types = {
            'Gender': types.SMALLINT(),
            'AgeGroup': types.SMALLINT(),
            'EducationLevel': types.SMALLINT(),
            'IncomeLevel': types.SMALLINT(),
            'EmploymentStatus': types.SMALLINT(),
            'MaritalStatus': types.SMALLINT(),
            'HomeOwnership': types.SMALLINT(),
            'GeneralHealth': types.SMALLINT(),
            'PhysicalHealth_DaysBad': types.INTEGER(),
            'MentalHealth_DaysBad': types.INTEGER(),
            'HealthLimitedDays': types.INTEGER(),
            'HasHealthPlan': types.SMALLINT(),
            'HasPersonalDoctor': types.SMALLINT(),
            'CouldNotAffordDoctor': types.SMALLINT(),
            'LastRoutineCheckup': types.SMALLINT(),
            'HighBloodPressure': types.SMALLINT(),
            'MedForBP': types.SMALLINT(),
            'CholesterolChecked': types.SMALLINT(),
            'HighCholesterol': types.SMALLINT(),
            'Diabetes': types.SMALLINT()
        }

        # --- Upload Cleaned Data ---
        upload_df_to_db_in_batches(df, chunksize=20000, table_name="staging_table", engine=staging_engine, dtype=sql_types)

        # --- Load Rules and Apply ---
        rules_df = pd.read_excel("Book 5.xlsx")
        rules_df.columns = rules_df.columns.str.strip().str.lower().str.replace('\n', '', regex=True).str.replace(' ', '_')

        if 'staging_column_name' not in rules_df.columns:
            raise KeyError("The column 'staging_column_name' is missing.")

        rules_df = rules_df.head(20)
        all_invalid_rows = pd.DataFrame()

        for _, row in rules_df.iterrows():
            col, rule = row['staging_column_name'], row.get('validation_rule')
            if pd.isna(col) or pd.isna(rule):
                continue
            condition = parse_rule_to_sql_condition(col, rule)
            if not condition:
                continue
            sql_query = f"SELECT * FROM staging_table WHERE {condition}"
            delete_query = f"DELETE FROM staging_table WHERE {condition}"
            invalid_rows = pd.read_sql_query(sql_query, staging_engine)
            if not invalid_rows.empty:
                invalid_rows['invalid_column'] = col
                invalid_rows['violated_rule'] = rule
                all_invalid_rows = pd.concat([all_invalid_rows, invalid_rows], ignore_index=True)
                with staging_engine.begin() as conn:
                    conn.execute(text(delete_query))

        if not all_invalid_rows.empty:
            all_invalid_rows.drop_duplicates(inplace=True)
            all_invalid_rows.to_csv("invalid_rows.csv", index=False)
            print("Invalid rows saved.")
        else:
            print("No invalid rows found.")

    except Exception as e:
        print(f"Staging pipeline failed: {e}")

if __name__ == "__main__":
    main()

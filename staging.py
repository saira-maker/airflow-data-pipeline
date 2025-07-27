import pandas as pd
import os
import re
import smtplib
from sqlalchemy import create_engine, text, types

# Rule Parsing Function
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

    return f"({column}) NOT IN ({values_str})"

# Batch upload
def upload_df_to_db_in_batches(df, chunksize, table_name, engine, dtype=None):
    total_rows = 0
    num_chunks = (len(df) + chunksize - 1) // chunksize
    for i in range(num_chunks):
        chunk = df.iloc[i * chunksize:(i + 1) * chunksize]
        chunk.to_sql(table_name, engine, if_exists='append' if i > 0 else 'replace', index=False, dtype=dtype)
        total_rows += len(chunk)
        print(f"Chunk {i+1}/{num_chunks}: loaded {len(chunk)} rows")
    print(f"Total rows uploaded: {total_rows}")

def main():
    try:
        username = 'postgres'
        password = 'postgre'  # Update this
        host = 'localhost'
        port = '5432'

        database_raw = 'raw_data'
        database_staging = 'staging_data'

        raw_engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database_raw}')
        raw_df = pd.read_sql_table('raw_table', raw_engine)
        raw_df.columns = [col.lower() for col in raw_df.columns]

        # Rename columns map
        rename_map = {
            'sex': 'Gender',
            'agegrp': 'AgeGroup',
            'educa': 'EducationLevel',
            'income2': 'IncomeLevel',
            'employ1': 'EmploymentStatus',
            'marital': 'MaritalStatus',
            'renthom1': 'HomeOwnership',
            'hlthpln1': 'HasHealthPlan',
            'persdoc2': 'HasPersonalDoctor',
            'medcost': 'CouldNotAffordDoctor',
            'checkup1': 'LastRoutineCheckup',
            'bp_high4': 'HighBloodPressure',
            'bpmeds': 'OnBloodPressureMeds',
            'cholchk': 'CholesterolChecked',
            'toldhi2': 'HighCholesterol',
            'diabete3': 'Diabetes'
        }

        # Filter columns present in raw_df and rename
        rename_map_filtered = {k: v for k, v in rename_map.items() if k in raw_df.columns}
        df = raw_df[list(rename_map_filtered.keys())].rename(columns=rename_map_filtered)

        # Drop empty columns and rows
        df.dropna(axis=1, how='all', inplace=True)
        df.dropna(axis=0, how='all', inplace=True)

        # Replace NaN with None for SQL compatibility
        df = df.where(pd.notnull(df), None)

        # Define SQL types
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

        # Adjust SMALLINT to INTEGER if needed
        SMALLINT_MAX = 32767
        for col, dtype in list(sql_types.items()):
            if isinstance(dtype, types.SMALLINT) and col in df.columns:
                max_val = df[col].max(skipna=True)
                if max_val is not None and max_val > SMALLINT_MAX:
                    sql_types[col] = types.INTEGER()
                    print(f"Column '{col}' datatype changed from SMALLINT to INTEGER due to max value {max_val}")

        # Upload cleaned data
        staging_engine = create_engine(f'postgresql://{username}:{password}@{host}:{port}/{database_staging}')
        chunksize = 20000
        print("Uploading cleaned data to staging table in batches...")
        upload_df_to_db_in_batches(df, chunksize, 'staging_table', staging_engine, dtype=sql_types)

        # Load validation rules
        rules_df = pd.read_excel("Book 5.xlsx")
        rules_df.columns = (
            rules_df.columns
            .str.strip()
            .str.lower()
            .str.replace('\n', '', regex=True)
            .str.replace('m', '_')
        )

        if 'staging_column_name' not in rules_df.columns:
            raise KeyError("The column 'staging_column_name' is missing from the validation rules file.")

        rules_df = rules_df.head(20)

        # Validate rows
        all_invalid_rows = pd.DataFrame()

        for idx, row in rules_df.iterrows():
            col = row['staging_column_name']
            rule = row.get('validation_rule', None)

            if pd.isna(col) or pd.isna(rule):
                continue

            condition = parse_rule_to_sql_condition(col, rule)
            if not condition:
                print(f"Skipping rule for '{col}': could not parse '{rule}'")
                continue

            sql_query = f"SELECT * FROM staging_table WHERE {condition};"
            delete_query = f"DELETE FROM staging_table WHERE {condition};"

            print(f"\nValidating column '{col}' with rule: '{rule}'")

            invalid_rows = pd.read_sql_query(sql_query, staging_engine)

            if invalid_rows.empty:
                print(f"All rows valid for '{col}'.")
            else:
                print(f"Found {len(invalid_rows)} invalid rows for '{col}'.")
                invalid_rows['invalid_column'] = col
                invalid_rows['violated_rule'] = rule
                all_invalid_rows = pd.concat([all_invalid_rows, invalid_rows], ignore_index=True)

            with staging_engine.begin() as conn:
                conn.execute(text(delete_query))
                print(f"Deleted {len(invalid_rows)} invalid rows from staging_table.")

        # Save invalid rows to CSV if any
        if not all_invalid_rows.empty:
            all_invalid_rows.drop_duplicates(inplace=True)
            all_invalid_rows.to_csv("invalid_rows.csv", index=False)
            print("\nInvalid rows saved to 'invalid_rows.csv'")
        else:
            print("\nNo invalid rows found or deleted.")

    except Exception as e:
        error_message = f"Staging data processing failed:\n\n{str(e)}"
        print(error_message)

# Run the main function
if __name__ == "__main__":
    main()

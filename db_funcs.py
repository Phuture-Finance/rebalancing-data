import sqlite3

db = (
    "../rebalancing_data_db.sqlite"
    if __name__ != "__main__"
    else "rebalancing_data_db.sqlite"
)


def create_connection(db_file):
    conn = None

    try:
        conn = sqlite3.connect(db_file, isolation_level=None)
    except Exception as e:
        print(e)
    return conn


def create_table(statement):
    conn = create_connection(db)
    try:
        connection = conn.cursor()
        connection.execute(statement)
        print("Table created successfully")
    except Exception as e:
        print(e)


def add_column(column, table_name):
    conn = create_connection(db).cursor()
    statement = f"ALTER TABLE {table_name} ADD {column} DOUBLE PRECISION"
    conn.execute(statement)


def drop_column(column, table_name):
    conn = create_connection(db).cursor()
    statement = f"ALTER TABLE {table_name} DROP COLUMN {column} "
    conn.execute(statement)


def does_column_exist(column, table_name):
    conn = create_connection(db).cursor()
    statement = f"SELECT length({column}) FROM {table_name}"
    try:
        conn.execute(statement)
        return True
    except:
        add_column(column, table_name)
        return True


def insert_values(row, columns, values, table_name):
    for column in columns:
        does_column_exist(column, table_name)
    conn = create_connection(db).cursor()
    columns = ",".join(columns)
    values = str(values).strip("[]").replace(" ", "")
    statement = f"""
    REPLACE INTO {table_name}(date,{columns})
    VALUES("{row}",{values})"""
    conn.execute(statement)


# Helper functions
def convert_to_sql_strings(string_list):
    mapping = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
    }
    converted_strings = []
    for i in string_list:
        if "-" in i:
            i = i.replace("-", "_")
        for number in list(mapping.keys()):
            if number in i:
                i = i.replace(number, mapping[number])
        converted_strings.append(i)
    return converted_strings


def convert_from_sql_strings(string_list):
    mapping = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }
    converted_strings = []
    for i in string_list:
        if "_" in i:
            i = i.replace("_", "-")
        for number in list(mapping.keys()):
            if number in i:
                i = i.replace(number,mapping[number])
        converted_strings.append(i)
    return converted_strings

def create_liquidity_table(name):
    sql_string = f"""CREATE TABLE IF NOT EXISTS {name} (
	date text PRIMARY KEY
);"""
    create_table(sql_string)

def create_benchmark_table(name):
    sql_string = f"""CREATE TABLE IF NOT EXISTS {name} (
	date text PRIMARY KEY
);"""
    create_table(sql_string)
    

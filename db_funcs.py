import sqlite3

db = "../asset_liquidities.sqlite" if __name__ != "__main__" else "asset_liquidities.sqlite"

def create_connection(db_file):
    conn = None

    try:
        conn = sqlite3.connect(db_file,isolation_level=None)
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

def add_column(column,table_name):
    conn =  create_connection(db).cursor()
    statement = f"ALTER TABLE {table_name} ADD \"{column}\" DOUBLE PRECISION"
    conn.execute(statement)

def drop_column(column,table_name):
    conn =  create_connection(db).cursor()
    statement = f"ALTER TABLE {table_name} DROP COLUMN {column} "
    conn.execute(statement)

def does_column_exist(column,table_name):
    conn =  create_connection(db).cursor()
    statement = f"SELECT length(\"{column}\") FROM {table_name}"
    print(statement)
    try:
        conn.execute(statement)
        return True
    except:
        add_column(column,table_name)
        return True
        
def insert_values(row,columns,values,table_name):
    for column in columns:
        does_column_exist(column,table_name)
    conn =  create_connection(db).cursor()
    columns = ",".join(columns)
    values = str(values).strip('[]').replace(' ','')
    statement = f"""
    REPLACE INTO {table_name}(date,\"{columns}\")
    VALUES('{row}',{values});"""
    conn.execute(statement)
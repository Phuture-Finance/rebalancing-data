import sqlite3

db = "asset_liquidities.sqlite"

def create_connection(db_file):
    conn = None

    try:
        conn = sqlite3.connect(db_file)
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

def add_column(id,table_name):
    conn =  create_connection(db).cursor()
    statement = f"ALTER TABLE {table_name} ADD {id} DOUBLE PRECISION"
    conn.execute(statement)

def drop_column(id,table_name):
    conn =  create_connection(db).cursor()
    statement = f"ALTER TABLE {table_name} DROP COLUMN {id} "
    conn.execute(statement)

def does_column_exist(id,table_name):
    conn =  create_connection(db).cursor()
    statement = f"SELECT length({id}) FROM {table_name}"
    try:
        conn.execute(statement)
        return True
    except:
        add_column(id,table_name)
        return True
        
def insert_values(id,value,table_name):
    conn =  create_connection(db).cursor()
    statement = f"INSERT INTO {table_name}(id,{id}) VALUES(1,{value});"
    conn.execute(statement)
    


       
create_table_sql = """ CREATE TABLE IF NOT EXISTS pdi_liquidities (
id INTEGER NULL PRIMARY KEY AUTOINCREMENT
);
"""

insert_values("Uniswap",0.03,"pdi_liquidities")
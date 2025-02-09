import mysql.connector

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',         # default MySQL username in XAMPP
        password='',         # default MySQL password in XAMPP (if set)
        database='user_database'  # the name of the database you created
    )

# Test the connection
if __name__ == "__main__":
    try:
        conn = get_db_connection()
        print("Connection successful")
        conn.close()
    except mysql.connector.Error as err:
        print(f"Error: {err}")

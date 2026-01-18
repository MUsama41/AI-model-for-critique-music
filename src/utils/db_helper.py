import psycopg2
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseClient:
    def __init__(self):
        self.dbname = os.getenv("dbname")
        self.user = os.getenv("user")
        self.password = os.getenv("password")
        self.host = os.getenv("host")
        self.port = os.getenv("port")

    def get_connection(self):
        try:
            conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            return conn
        except Exception as e:
            print(f"Database connection error: {e}")
            return None

    def check_existing_rating(self, features, table_name):
        conn = self.get_connection()
        if not conn: return None
        
        cur = conn.cursor()
        query = f"SELECT spectral_rolloff, zero_crossing_rate, spectral_bandwidth, spectral_centroid, spectral_contrast, mfcc, chroma, popularity FROM {table_name}"
        cur.execute(query)
        rows = cur.fetchall()
        
        tolerance = 1e-5
        relevant_columns = ['spectral_rolloff', 'zero_crossing_rate', 'spectral_bandwidth', 'spectral_centroid', 'spectral_contrast', 'mfcc', 'chroma']
        features_values = features[relevant_columns].values[0]

        label = None
        for row in rows:
            row_values = row[:-1]
            if np.all(np.isclose(features_values, row_values, atol=tolerance)):
                label = row[-1]
                break
        
        cur.close()
        conn.close()
        return label

    def store_feedback(self, features, feedback):
        conn = self.get_connection()
        if not conn: return
        
        cur = conn.cursor()
        for table in ["song_feedback", "retrain_db"]:
            existing = self.check_existing_rating(features, table)
            if existing is not None:
                query = f"""
                UPDATE {table} SET popularity = %s WHERE 
                spectral_rolloff = %s AND zero_crossing_rate = %s AND spectral_bandwidth = %s AND 
                spectral_centroid = %s AND spectral_contrast = %s AND mfcc = %s AND chroma = %s
                """
                cur.execute(query, [feedback] + list(features.values[0]))
            else:
                query = f"INSERT INTO {table} (spectral_rolloff, zero_crossing_rate, spectral_bandwidth, spectral_centroid, spectral_contrast, mfcc, chroma, popularity) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                cur.execute(query, list(features.values[0]) + [feedback])
        
        conn.commit()
        cur.close()
        conn.close()

    def get_training_data(self):
        conn = self.get_connection()
        if not conn: return pd.DataFrame()
        query = "SELECT * FROM big_data;"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def migrate_retrain_data(self, min_count=1):
        conn = self.get_connection()
        if not conn: return False
        
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM retrain_db;")
        if cur.fetchone()[0] < min_count:
            cur.close()
            conn.close()
            return False
            
        columns = ['spectral_rolloff', 'zero_crossing_rate', 'spectral_bandwidth', 'spectral_centroid', 'spectral_contrast', 'mfcc', 'chroma', 'popularity']
        cols_str = ", ".join(columns)
        cur.execute(f"INSERT INTO big_data ({cols_str}) SELECT {cols_str} FROM retrain_db;")
        cur.execute("TRUNCATE TABLE retrain_db;")
        conn.commit()
        cur.close()
        conn.close()
        return True

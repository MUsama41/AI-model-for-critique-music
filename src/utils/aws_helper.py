import boto3
import os
from dotenv import load_dotenv

load_dotenv()

class AWSClient:
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )

    def download_file(self, song_name, bucket_name, bucket_region):
        local_file_path = os.path.join('temp', song_name)
        os.makedirs('temp', exist_ok=True)
        
        try:
            self.s3.download_file(bucket_name, song_name, local_file_path)
            return local_file_path
        except Exception as e:
            raise Exception(f"S3 download failed: {str(e)}")

    @staticmethod
    def remove_temp_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

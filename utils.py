import os

def load_credentials():
    with open('secrets.txt') as f:
        for line in f:
            (api, cred) = line.split(" ")
            if not os.getenv(api):
                os.environ[api] = cred.strip()
                
                
if __name__ == "__main__":
    load_credentials()
    print("Credentials loaded successfully.")
            
        
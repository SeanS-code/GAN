from model.model import gan_model
from gans.data import splits

def data():
    splits()

def main():
    gan_model()

if __name__ == "__main__":
    main()
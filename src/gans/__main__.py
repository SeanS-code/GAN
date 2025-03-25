from gans.self_data import splits, load_data
from model.model import gan_model

def data():
    splits()

def main():
    load_data(True)

if __name__ == "__main__":
    main()
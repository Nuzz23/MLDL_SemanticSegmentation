from models.deeplabv2.deeplabv2 import DeepLabV2

def train():
    pass





def validate():
    pass




def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
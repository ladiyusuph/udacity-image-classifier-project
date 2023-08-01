# Importing required libraries
import argparse

def training_args():
    # Creating Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Creating command line arguments
    parser.add_argument('--data_dir', type=str, default='flowers/', 
                        help='path to image folder')
    parser.add_argument('--save_dir', type=str, default='save_directory', 
                        help='path to save trained models')
    parser.add_argument('--arch', type=str, default = 'vgg19',
                        help='model architecture')
    parser.add_argument('--learning_rate', type=float, default = 0.001,
                        help='learning rate of the model')
    parser.add_argument('--epochs', type=int, default = 5,
                        help='the number of epochs for the training')
    parser.add_argument('--hidden_units1', type=int, default = 4096,
                        help='hidden units for the model')
    parser.add_argument('--hidden_units2', type=int, default = 256,
                        help='hidden units for the model')
    parser.add_argument('--gpu', type=str, default = "gpu",
                        help='Use gpu for training model')
    
    return parser.parse_args()


def predict_args():
    # Creating Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Creating command line arguments
    parser.add_argument('--data_dir', type=str, default='flowers/', 
                        help='path to image folder')
    parser.add_argument('--save_dir', type=str, default='save_directory', 
                        help='path to saved models')
    parser.add_argument('--arch', type=str, default = 'vgg19',
                        help='model architecture')
    parser.add_argument('--category_names', type=str, default = 'cat_to_name.json', 
                        help='json file of names of flowers')
    parser.add_argument('--top_k', type=int, default = 5,
                        help='top class probability for prediction')
    parser.add_argument('--gpu', type=str, default = "gpu",
                        help='Use gpu for model prediction')
    parser.add_argument('--image_path', type=str, default = "flowers/test/43/image_02394.jpg",
                        help='Path to image for prediction')
    
    return parser.parse_args()
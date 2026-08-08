from datasets import load_dataset

def main():
    # 1. Load the dataset
    print("Downloading and loading the dataset...")
    ds = load_dataset("korexyz/celeba-hq-256x256")
    
    # 2. Inspect the DatasetDict structure
    # This will show you the available splits (like 'train', 'test', 'validation')
    print("\nDataset structure:")
    print(ds)
    
    # 3. Access a specific split
    # Hugging Face datasets are usually stored in a dictionary-like structure
    train_data = ds['train']
    print(f"\nNumber of training examples: {len(train_data)}")
    
    # 4. Access a specific item
    first_item = train_data[0]
    print("\nFirst item data structure:")
    print(first_item)
    
    # 5. View the image
    # Assuming the column containing the image is named 'image' (standard for HF)
    image = first_item['image']
    image.show() # Opens the image in your default image viewer

if __name__ == "__main__":
    main()
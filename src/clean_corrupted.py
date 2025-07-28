import os
from PIL import Image
from tqdm import tqdm

def clean_corrupt_images(directory):
    corrupt_files = []
    for root, _, files in os.walk(directory):
        for file in tqdm(files, desc=f"Scanning {os.path.basename(root)}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    # Dosyayı aç ve bağımsız olarak doğrula
                    with Image.open(file_path) as img:
                        img.verify()  # Temel doğrulama
                    
                    # Ek kontrol için yeniden aç
                    with Image.open(file_path) as img:
                        img.load()  # Tüm veriyi yükle
                        img.transpose(Image.FLIP_LEFT_RIGHT)  # İşlem testi
                except Exception as e:
                    corrupt_files.append(file_path)
                    os.remove(file_path)
                    print(f"Deleted corrupt file: {os.path.basename(file_path)} - Error: {str(e)}")
    return corrupt_files

if __name__ == "__main__":
    dataset_path = r'C:\Users\Пользователь\Desktop\pbid\data\train'
    print(f"Starting corruption check in: {dataset_path}")
    corrupt_files = clean_corrupt_images(dataset_path)
    print(f"\nTotal deleted corrupt files: {len(corrupt_files)}")
    if corrupt_files:
        print("Sample deleted files:")
        for f in corrupt_files[:5]:  # İlk 5 bozuk dosyayı göster
            print(f" - {os.path.basename(f)}")
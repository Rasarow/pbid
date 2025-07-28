import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

model_path = os.path.join(r'C:\Users\Пользователь\Desktop\pbid\models', 'best_model.h5')
model = load_model(model_path)

class_names = [ 
    'Abyssinian',                         # 0
    'American Bobtail',                   # 1
    'American Shorthair',                 # 2
    'Bengal',                             # 3
    'Birman',                             # 4
    'Bombay',                             # 5
    'British_Shorthair',                  # 6
    'Egyptian_Mau',                       # 7
    'Maine_Coon',                         # 8
    'Oriental Short Hair',                # 9
    'Persian',                            # 10
    'Ragdoll',                            # 11
    'Russian_Blue',                       # 12
    'Siamese',                            # 13
    'Sphynx',                             # 14
    'Tuxedo',                             # 15
    'n02085620-Chihuahua',                # 16
    'n02085782-Japanese_spaniel',         # 17
    'n02085936-Maltese_dog',              # 18
    'n02086079-Pekinese',                 # 19
    'n02086240-Shih-Tzu',                 # 20
    'n02086646-Blenheim_spaniel',         # 21
    'n02086910-papillon',                 # 22
    'n02087046-toy_terrier',              # 23
    'n02087394-Rhodesian_ridgeback',      # 24
    'n02088094-Afghan_hound',             # 25
    'n02088238-basset',                   # 26
    'n02088364-beagle',                   # 27
    'n02088466-bloodhound',               # 28
    'n02088632-bluetick',                 # 29
    'n02089078-black-and-tan_coonhound',  # 30
    'n02089867-Walker_hound',             # 31
    'n02089973-English_foxhound',         # 32
    'n02090379-redbone',                  # 33
    'n02090622-borzoi',                   # 34
    'n02090721-Irish_wolfhound',          # 35
    'n02091032-Italian_greyhound',        # 36
    'n02091134-whippet',                  # 37
    'n02091244-Ibizan_hound',             # 38
    'n02091467-Norwegian_elkhound',       # 39
    'n02091635-otterhound',               # 40
    'n02091831-Saluki',                   # 41
    'n02092002-Scottish_deerhound',       # 42
    'n02092339-Weimaraner',               # 43
    'n02093256-Staffordshire_bullterrier',# 44
    'n02093428-American_Staffordshire_terrier', # 45
    'n02093647-Bedlington_terrier',       # 46
    'n02093754-Border_terrier',           # 47
    'n02093859-Kerry_blue_terrier',       # 48
    'n02093991-Irish_terrier',            # 49
    'n02094114-Norfolk_terrier',          # 50
    'n02094258-Norwich_terrier',          # 51
    'n02094433-Yorkshire_terrier',        # 52
    'n02095314-wire-haired_fox_terrier',  # 53
    'n02095570-Lakeland_terrier',         # 54
    'n02095889-Sealyham_terrier',         # 55
    'n02096051-Airedale',                 # 56
    'n02096177-cairn',                    # 57
    'n02096294-Australian_terrier',       # 58
    'n02096437-Dandie_Dinmont',           # 59
    'n02096585-Boston_bull',              # 60
    'n02097047-miniature_schnauzer',      # 61
    'n02097130-giant_schnauzer',          # 62
    'n02097209-standard_schnauzer',       # 63
    'n02097298-Scotch_terrier',           # 64
    'n02097474-Tibetan_terrier',          # 65
    'n02097658-silky_terrier',            # 66
    'n02098105-soft-coated_wheaten_terrier', # 67
    'n02098286-West_Highland_white_terrier', # 68
    'n02098413-Lhasa',                    # 69
    'n02099267-flat-coated_retriever',    # 70
    'n02099429-curly-coated_retriever',   # 71
    'n02099601-golden_retriever',         # 72
    'n02099712-Labrador_retriever',       # 73
    'n02099849-Chesapeake_Bay_retriever', # 74
    'n02100236-German_short-haired_pointer', # 75
    'n02100583-vizsla',                   # 76
    'n02100735-English_setter',           # 77
    'n02100877-Irish_setter',             # 78
    'n02101006-Gordon_setter',            # 79
    'n02101388-Brittany_spaniel',         # 80
    'n02101556-clumber',                  # 81
    'n02102040-English_springer',         # 82
    'n02102177-Welsh_springer_spaniel',   # 83
    'n02102318-cocker_spaniel',           # 84
    'n02102480-Sussex_spaniel',           # 85
    'n02102973-Irish_water_spaniel',      # 86
    'n02104029-kuvasz',                   # 87
    'n02104365-schipperke',               # 88
    'n02105056-groenendael',              # 89
    'n02105162-malinois',                 # 90
    'n02105251-briard',                   # 91
    'n02105412-kelpie',                   # 92
    'n02105505-komondor',                 # 93
    'n02105641-Old_English_sheepdog',     # 94
    'n02105855-Shetland_sheepdog',        # 95
    'n02106030-collie',                   # 96
    'n02106166-Border_collie',            # 97
    'n02106382-Bouvier_des_Flandres',     # 98
    'n02106550-Rottweiler',               # 99
    'n02106662-German_shepherd',          # 100
    'n02107142-Doberman',                 # 101
    'n02107312-miniature_pinscher',       # 102
    'n02107574-Greater_Swiss_Mountain_dog', # 103
    'n02107683-Bernese_mountain_dog',     # 104
    'n02107908-Appenzeller',              # 105
    'n02108000-EntleBucher',              # 106
    'n02108089-boxer',                    # 107
    'n02108422-bull_mastiff',             # 108
    'n02108551-Tibetan_mastiff',       # 109
    'n02108915-French_bulldog',        # 110
    'n02109047-Great_Dane',            # 111
    'n02109525-Saint_Bernard',         # 112
    'n02109961-Eskimo_dog',            # 113
    'n02110063-Malamute',              # 114
    'n02110185-Siberian_husky',        # 115
    'n02110341-dalmatian',             # 116
    'n02110627-affenpinscher',         # 117
    'n02110806-basenji',               # 118
    'n02110958-pug',                   # 119
    'n02111129-Leonberg',              # 120
    'n02111277-Newfoundland',          # 121
    'n02111500-Great_Pyrenees',        # 122
    'n02111889-Samoyed',               # 123
    'n02112018-Pomeranian',            # 124
    'n02112137-Chow_chow',             # 125
    'n02112350-keeshond',              # 126
    'n02112706-Brabancon_griffon',     # 127
    'n02113023-Pembroke',              # 128
    'n02113186-Cardigan',              # 129
    'n02113624-toy_poodle',            # 130
    'n02113712-miniature_poodle',      # 131
    'n02113799-standard_poodle',       # 132
    'n02113978-Mexican_hairless',      # 133
    'n02114367-dingo',                 # 134
    'n02114548-dhole',                 # 135
    ]

def predict_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
 
        print("\nEstimated results:")
        top10_idx = np.argsort(predictions[0])[-10:][::-1]  
        for idx in top10_idx:
            print(f"{class_names[idx]}: %{predictions[0][idx] * 100:.2f}")
        
        print(f"\n→ Highest estimated result: {class_names[predicted_class_idx]} (%{predictions[0][predicted_class_idx] * 100:.2f})")
        return class_names[predicted_class_idx]
    
    except Exception as e:
        print(f"An error occured: {str(e)}")
        return None

test_image_path = r"C:\Users\Пользователь\Desktop\shutterstock_2092702819.jpg"

if os.path.exists(test_image_path):
    predicted_class = predict_image(test_image_path)
else:
    print(f"File not found - {test_image_path}")
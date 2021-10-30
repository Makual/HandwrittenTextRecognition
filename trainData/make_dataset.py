import os
from PIL import Image
import numpy as np

result = [[],[]]



folders = os.listdir('base_1DIG')

for folder in folders:
    for image in os.listdir('base_1DIG/'+folder):
        img = Image.open('base_1DIG/'+folder+'/'+image)
        img = img.resize((64,64))

        result[0].append(np.array(img))
        result[1].append(folder)
    print('new')

folders = os.listdir('base_2DIG')

for folder in folders:
    for image in os.listdir('base_2DIG/'+folder):
        img = Image.open('base_2DIG/'+folder+'/'+image)
        img = img.resize((64,64))

        result[0].append(np.array(img))
        result[1].append(folder)
    print('new')

print(len(result[0]))


save_images = np.array(result[0])
save_letters = np.array(result[1])

np.save('save64_X_D.npy',save_images)
np.save('save64_Y_D.npy',save_letters)

#a = np.load('save_1Y.npy')

#print(a[56])

    
#a = Image.fromarray(a[65])
#a.show()


        
    
     

import telebot
import cv2
import numpy as np
import pandas as pd
import keras
from autocorrect import Speller
import difflib
import lib


spell = Speller(lang='ru')


token = '2062502044:AAHswmGbZRJIEVa5xOkdxyxW-kcE07ycmN4'


def image2text(img):
  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

  coords = lib.attributeSort(lib.get_LettersCoords(img), 'x')
  spaces = lib.get_spaces(coords, 2, False)
  print(spaces)
  letters, row = lib.predictRow(img,coords,(64,64),30)

  print(letters)

  tokens = lib.makeTokens(letters, row, spaces)

  text = ''

  for token in tokens:
    text += lib.get_bestCandidate(token) + ' '

  return text[:len(text)-1]


print('start')

bot = telebot.TeleBot(token)

@bot.message_handler(content_types=["text"])
def start(message):
    if message.text=='/start':
        bot.send_message(message.chat.id, 'Для распознавания сделайте фото текста')

@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    bot.reply_to(message, "Подождите немного.")
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    
    src = 'file.png'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
 
    img = cv2.imread('file.png')
    
    text = image2text(img)

    bot.send_message(message.chat.id, text)

    
    
 
if __name__ == '__main__':
     bot.infinity_polling()




import io
import os
import cv2
import numpy as np
import telebot
from PIL import Image

bot = telebot.TeleBot('1698393530:AAEowFlwIfNegtbvKWD8p8x-QkF-OzOc0Mw')

@bot.message_handler(commands=['start'])
def strt_com(message):
    bot.send_message(message.chat.id, "Скиньте фото")

@bot.message_handler(content_types=['photo'])
def docs_photo(message):
    file_info = bot.get_file(message.photo[len(message.photo) - 1].file_id)
    downloaded_file = bot.download_file('photo.png')

    src = 'photo.png'
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.reply_to(message, "Ждите")

cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

recognizer = cv2.createLBPHFaceRecognizer(1, 8, 8, 8, 123)

def get_images(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    images = []
    labels = []
    for image_path in image_paths:
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        subject_number = int(os.path.split(image_path)[1].split(".")[0])
        faces = faceCascade.detectMultiScale(
            image, scaleFactor=1.5, minNeighbors=6, minSize=(30, 30))
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(subject_number)
            cv2.imshow("", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    return images, labels

path = './yalefaces'
images, labels = get_images(path)
cv2.destroyAllWindows()
recognizer.train(images, np.array(labels))
image_paths = [os.path.join(path, f) for f in os.listdir(path)]

for image_path in image_paths:
    gray = Image.open(image_path).convert('L')
    image = np.array(gray, 'uint8')
    faces = faceCascade.detectMultiScale(
        image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])
        number_actual = int(os.path.split(image_path)[1].split(".")[0])
        if number_actual == number_predicted:
            print("Распознанно {} {}".format(number_actual, conf))
        else:
            print("Не распознанно {} {}".format(
                number_actual, number_predicted))
        cv2.imshow("Распознавание", image[y: y + h, x: x + w])
        cv2.waitKey(1000)

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
recognizer = cv2.createLBPHFaceRecognizer(1, 8, 8, 8, 123)
bot.polling(none_stop=True)


import pyrebase
from datetime import date, datetime

config = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "YOUR_AUTH_DOMAIN",
    "databaseURL": "YOUR_DATABASE_URL",
    "projectId": "YOUR_PROJECT_ID",
    "storageBucket": "YOUR_STORAGE_BUCKET",
    "messagingSenderId": "YOUR_MESSAGING_SENDER_ID",
    "appId": "YOUR_APP_ID",
    "serviceAccount": "serviceAccount.json" 
}

firebase = pyrebase.initialize_app(config)
storage = firebase.storage()
database = firebase.database()
storage.child()

### 1. download image from database
# all_files = storage.child().list_files()
# for file in all_files:            
#     try:
#         print(file.name) 
#         storage.download(file.name, file.name)
#     except:    
#         print('Download Failed')

### 2. set data to firebase
# set data in given format only
# database.child()
# data = {"Data": 
#                 {"category1":"price1",
#                  "category2":"price2"}
#        }
# database.push(data)

### 3. get val from database
# price = database.child("Receipt_Images").get()
# data = price.val() 
# print(data)

### 4. number of days
# current_time = datetime.now()
# d0 = date(current_time.year, 6, 1)
# d1 = date(current_time.year, current_time.month, current_time.day)
# delta = d1 - d0
# data2 = {"analytics":
#          {  
#             "months":(delta.days)//30,
#             "days":delta.days,
#             "total":"total",
#         }}
# database.child()
# database.push(data2)



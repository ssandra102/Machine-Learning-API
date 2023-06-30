# Machine-Learning-API
Flask API developed using Python Flask, to extract texts from Indian itemized receipts. The extracted text is classified into categories belonging to one of the 21 categories(see 'data' dictinonary in main.py).<br>
The cumulative sum of categorised items along with their respective category is pushed to Firebase Realtime Database. Also, the receipt image is fetched from Firebase Storage. <br>

## Files
1. main.py - The API is written in this file. Run the file using the command:
```
python main.py
## or
python3 main.py
```
<br>The frontend is a webpage with the text "Hello World". <br>
2. Fetch_images.py - contains configuration details of firebase, that you will get by creating a new project in firebase.<br>
3. serviceAccount.json -  <br>
4. requirements.txt - contains the libraries used for the project.<br>
5. SVC_model.pkl - a pickle file used for categorising, receipt items. It is a SVC model, used for multi-text classification with 3 pre-processing steps done on the text. They are coded as a pipeline with the following functions: removing stopwords, porter stemming, and tf-idf vectoriser.<br>
6. Categorization.ipynb - notebook with all the steps used to develop SVM model i.e SVC_model.pkl.<br>

## Dataset
DATA1.csv - 11179 rows with 3 columns of Indian product desccription, sub category, and category.


 

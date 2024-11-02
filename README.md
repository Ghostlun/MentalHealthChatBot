# MyChatBotCreation Project

1. **Mental Health Synthetic Dataset** (loaded as `mental_df`)
2. **Mental Health Counsel Chatbot Dataset** (loaded as `counsel_df`)

## Mental Health Synthetic Dataset Analysis
## Dataset Overview

The **Mental Health Synthetic Dataset** (`mental_df`) contains the following columns:

- `User ID`, `Age`, `Gender`, `Symptoms`, `Duration (weeks)`, `Previous Diagnosis`, `Therapy History`, `Medication`, `Diagnosis / Condition`, `Suggested Therapy`, `Self-care Advice`, `Urgency Level`, `Mood`, `Stress Level`

This dataset is used primarily for building diagnostic and recommendation models based on user symptoms and demographic data.

## Step 1: Encoding Data for Model Training

Since the original data in `mental_df` was mostly in string format, I encoded categorical variables to numerical values for model compatibility. This process involved:

1. **Diagnosis Encoding**: Mapping diagnosis types to numerical labels.
2. **Symptom Encoding**: Converting symptom descriptions to numerical labels.
3. **Self-care and Therapy Advice Encoding**: Encoding recommendations for self-care and therapy.

```python
def build_self_test_self_care_advice():
    le_diagnosis = LabelEncoder()
    le_symptoms = LabelEncoder()
    le_self_care = LabelEncoder()
    le_therapy = LabelEncoder()

    # Encoding columns
    mental_df['Diagnosis_encoded'] = le_diagnosis.fit_transform(mental_df['Diagnosis'])
    mental_df['Symptoms_encoded'] = le_symptoms.fit_transform(mental_df['Symptoms'])
    mental_df['Self_Care_Advice_encoded'] = le_self_care.fit_transform(mental_df['Self_Care_Advice'])
    mental_df['Suggested_Therapy_encoded'] = le_therapy.fit_transform(mental_df['Suggested_Therapy'])

    # Model training setup
    X = mental_df[['Diagnosis_encoded', 'Symptoms_encoded']]
    y_self_care = mental_df['Self_Care_Advice_encoded']
    y_therapy = mental_df['Suggested_Therapy_encoded']

    X_train, X_test, y_self_care_train, y_self_care_test, y_therapy_train, y_therapy_test = train_test_split(X, y_self_care, y_therapy, test_size=0.2, random_state=42)

    # Train models
    model_self_care = RandomForestClassifier()
    model_therapy = RandomForestClassifier()

    model_self_care.fit(X_train, y_self_care_train)
    model_therapy.fit(X_train, y_therapy_train)

    # Predictions
    self_care_pred = model_self_care.predict(X_test)
    therapy_pred = model_therapy.predict(X_test)
```

### **Benefits of Encoding**

Encoding data allows models like RandomForest to understand the categorical relationships between features. By converting text-based data into numerical values, the model can learn patterns effectively without being hindered by incompatible data types.

### **Initial Model Results**

- **Accuracy**: Initial self-care and therapy advice prediction accuracy was low (~20%).
- **Insight**: The low accuracy indicated that `mental_df` might lack strong patterns for recommendations due to limited data diversity in these fields.

## Step 2: Improved Model for Diagnosis Prediction

Given the low accuracy of self-care and therapy advice predictions, I pivoted to focus on diagnosing users based on a broader range of features.

### **Data Grouping and Re-Mapping**

To improve prediction accuracy, I grouped diagnoses and previous diagnoses into higher-level categories:

```python
def group_diagnosis(row):
    if row['Diagnosis'] in ['Panic Disorder', 'Anxiety']:
        return 'Anxiety Disorders'
    elif row['Diagnosis'] in ['Depression', 'Burnout']:
        return 'Mood Disorders'
    elif row['Diagnosis'] == 'Stress':
        return 'Stress-Related Disorders'

def group_prev_diagnosis(row):
    if row['Prev_Diagnosis'] in ['Panic Disorder', 'Anxiety', 'OCD']:
        return 'Anxiety Disorders'
    elif row['Prev_Diagnosis'] in ['Depression', 'Bipolar Disorder']:
        return 'Mood Disorders'
    elif row['Prev_Diagnosis'] in ['Stress', 'PTSD']:
        return 'Stress-Related Disorders'
    else:
        return ''
```

### **Label Encoding for Diagnosis and Related Fields**

```python
def improved_test_reports_diagnosis(mental_df):
    mental_df["Diagnosis_Group"] = mental_df.apply(group_diagnosis, axis=1)
    mental_df['Prev_Diagnosis_Group'] = mental_df.apply(group_prev_diagnosis, axis=1)
    mental_df["Re_Gender"] = mental_df.apply(re_map_gender, axis=1)
    mental_df["Urgency_Level"] = mental_df.apply(re_map_urgency_level, axis=1)

    le_diagnosis_group = LabelEncoder()
    le_prev_diagnosis_group = LabelEncoder()
    le_symptoms = LabelEncoder()

    mental_df['Diagnosis_Group_encoded'] = le_diagnosis_group.fit_transform(mental_df['Diagnosis_Group'])
    mental_df['Prev_Diagnosis_Group_encoded'] = le_prev_diagnosis_group.fit_transform(mental_df['Prev_Diagnosis'])
    mental_df['Symptoms_encoded'] = le_symptoms.fit_transform(mental_df['Symptoms'])

    # Train Data Preparation
    X = mental_df[['Age', 'Symptoms_encoded', "Re_Gender", "Prev_Diagnosis_Group_encoded", "Duration", "Stress_Level", "Urgency_Level"]]
    y_diagnosis = mental_df['Diagnosis_Group_encoded']

    X_train, X_test, y_diagnosis_train, y_diagnosis_test = train_test_split(X, y_diagnosis, test_size=0.2, random_state=42)

    # Model Training
    model_diagnosis = RandomForestClassifier()
    model_diagnosis.fit(X_train, y_diagnosis_train)

    # Predictions and Evaluation
    diagnos_pred = model_diagnosis.predict(X_test)
    print("Diagnosis Group Classification Report:")
    diagnosis_report = classification_report(y_diagnosis_test, diagnos_pred, target_names=le_diagnosis_group.classes_)
    print(diagnosis_report)

    return model_diagnosis, le_diagnosis_group, diagnosis_report
```

### **Benefits of Grouping Diagnoses**

Grouping diagnoses reduces data sparsity and increases the likelihood of identifying meaningful patterns by clustering related conditions into broader categories (e.g., Anxiety, Mood Disorders). This approach improves model performance and provides a more generalized diagnosis output, which can still guide users effectively.

### **Model Results**

- **Accuracy**: Diagnosis model accuracy improved to around 40%.
- **Benefit**: With broader categories, the model gains better predictive power by leveraging the common characteristics within each diagnosis group.

## Summary of Benefits and Model Use

- **Label Encoding**: Necessary for model compatibility and to represent categorical data numerically.
- **Grouping and Re-Mapping**: Helped improve the prediction by consolidating data into categories that reflect real-world diagnoses.
- **RandomForestClassifier**: Offers robustness and flexibility with categorical data, making it an ideal choice for the diagnosis model.

With this approach, the chatbot can now use user-provided information to predict and suggest diagnosis groups, such as:
- **Anxiety Disorders**
- **Stress-Related Disorders**
- **Mood Disorders**

This structured approach ensures that the chatbot offers relevant diagnoses and recommendations based on the user's input, even with the limited dataset accuracy constraints.

## Counsel Dataset Analysis
## Dataset Overview

The **Mental Health Counsel Chatbot Dataset** (`counsel_df`) initially included the following columns:

- `questionID`, `questionTitle`, `questionText`, `questionUrl`, `topics`, `therapistName`, `therapistUrl`, `answerText`, `upvotes`

For the chatbot model, I focused on the columns: `questionTitle`, `questionText`, `topics`, and `answerText`. These fields are essential for building context around mental health topics, user inquiries, and appropriate responses.

## Step 1: Adding Diagnoses to Topics

Since `counsel_df` did not include direct diagnoses, I needed to add a diagnosis column (`re_diagnosis`) to categorize questions based on their topics. Initially, I mapped keywords in `topics` to diagnoses. For example:

- `Panic` → Panic Disorder
- `Depression` → Depression
- `Stress` → Stress-Related Disorders

However, this simple keyword matching method was limited. To improve, I:

1. **Used NLP Techniques**: I leveraged the **NLTK** library to find synonyms and related words for each keyword, enhancing the accuracy of topic-based diagnosis categorization.
2. **Analyzed Word Frequencies**: By calculating word frequencies, I identified which keywords and related terms occurred most often, helping refine the diagnosis mapping.

### Word Frequency Analysis Code

```python
from collections import Counter

def get_word_frequencies(counsel_df):
    counsel_df = counsel_df[['questionText', 'topics', 'answerText']]
    all_words = ' '.join(counsel_df['topics'].astype(str)).replace(',', '').split()
    word_count = Counter(all_words)

    stress_count = 0
    depression_count = 0
    disorder_count = 0
    anxiety_count = 0
    burn_out_count = 0

    for word, count in word_count.items():
        if word.__contains__("Stress"):
            stress_count += count
        elif word.__contains__("Depression"):
            depression_count += count
        elif word.__contains__("Disorder"):
            disorder_count += count
        elif word.__contains__("Anxiety"):
            anxiety_count += count
        elif word.__contains__("Burnout"):
            burn_out_count += count

    print("Total relevant word count:", stress_count + depression_count + disorder_count + anxiety_count + burn_out_count)
```

### **Benefit of Word Frequency Analysis**

This analysis helps identify the prevalence of certain mental health terms, allowing for more accurate topic-diagnosis mapping. By understanding term frequency, we can create a foundation for better text-based diagnostics.

## Step 2: Building a Diagnosis Prediction Model

Since only a subset of the `counsel_df` topics could be directly mapped to diagnoses, I trained a machine learning model to predict diagnoses for the remaining data. 

### Model Training with TF-IDF and Logistic Regression

To predict diagnoses for `counsel_df`:

1. **Vectorization with TF-IDF**: I applied **TF-IDF Vectorization** to the `topics` column. TF-IDF assigns weights to words based on their frequency within individual entries, making it ideal for text classification in this context.
2. **Logistic Regression Model**: I used a logistic regression model trained on the TF-IDF vectors to predict diagnoses.

#### Code for Creating the Diagnosis Model

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def create_diagnosis_model(train_data, target_column='re_diagnosis'):
    # Extract text and target columns
    X_train = train_data['topics']
    y_train = train_data[target_column]

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)

    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Evaluate model accuracy on the training set
    X_train_pred = model.predict(X_train_tfidf)
    accuracy = accuracy_score(y_train, X_train_pred)
    print(f"Training Accuracy: {accuracy}")

    return model, tfidf
```

### **Benefits of TF-IDF and Logistic Regression**

- **TF-IDF**: Effectively captures important terms in the text, allowing the model to focus on relevant words for each diagnosis category.
- **Logistic Regression**: Suitable for classification tasks and performs well with text-based data, especially when combined with TF-IDF features.

## Step 3: Predicting Missing Diagnoses

With the trained model, I filled in missing diagnoses in `counsel_df` where `re_diagnosis` was initially empty.

```python
def predict_missing_diagnoses(df, model, tfidf, target_column='re_diagnosis'):
    # Filter rows without a diagnosis
    df_test = df[df[target_column].isna()]
    X_test = df_test['topics'].fillna('').str.strip()
    X_test_tfidf = tfidf.transform(X_test)

    # Predict missing diagnoses
    predictions = model.predict(X_test_tfidf)

    # Assign predictions back to the DataFrame
    df.loc[df[target_column].isna(), target_column] = predictions

    return df
```

### **Benefits of Predicting Missing Diagnoses**

This step ensures that all entries in `counsel_df` have an assigned diagnosis, allowing for a more comprehensive and accurate chatbot response system. By using a machine learning model to fill in gaps, the data becomes more uniform and reliable for downstream processes.

## Summary of Approach and Benefits

1. **Enhanced Topic Mapping**: Using NLP to identify synonyms and related terms improved diagnosis mapping accuracy.
2. **TF-IDF with Logistic Regression**: Provided a reliable way to predict diagnoses, leveraging word importance within each topic.
3. **Comprehensive Diagnosis Assignment**: Ensured every entry in `counsel_df` had a diagnosis, creating a robust dataset for chatbot training and response accuracy.

This approach makes it possible to classify mental health-related inquiries based on topics, allowing the chatbot to provide accurate diagnoses and recommendations.
```

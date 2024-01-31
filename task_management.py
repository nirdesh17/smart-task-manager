import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model and vectorizer
def load_model_and_vectorizer(model_path='trained_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Load dynamic task file or create a new one if it doesn't exist
def load_dynamic_task_file(file_path='task_list.csv'):
    try:
        tasks_df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Create an empty DataFrame if the file doesn't exist
        tasks_df = pd.DataFrame(columns=['description', 'priority'])
    return tasks_df

# Save tasks to the dynamic task file
def save_dynamic_task_file(tasks_df, file_path='task_list.csv'):
    tasks_df.to_csv(file_path, index=False)

# Function to predict task priority using the trained model and vectorizer
def predict_priority(task_description, model, vectorizer):
    task_description_tfidf = vectorizer.transform([task_description])
    predicted_priority = model.predict(task_description_tfidf)
    return predicted_priority[0]

# Function to add a new task and assign priority
def add_task(task_description, tasks_df, model, vectorizer):
    priority = predict_priority(task_description, model, vectorizer)
    new_task = {'description': task_description, 'priority': priority}
    if(new_task==""):
        return tasks_df  # If no valid description is provided, don't add anything
    tasks_df = pd.concat([tasks_df, pd.DataFrame(new_task, index=[0])], ignore_index=True)
    save_dynamic_task_file(tasks_df)
    return tasks_df

# Function to remove a task by description
def remove_task(task_description, tasks_df):
    tasks_df = tasks_df[tasks_df['description'] != task_description]
    save_dynamic_task_file(tasks_df)
    return tasks_df

# Function to list all tasks (without printing priority and serial number)
def list_tasks(tasks_df):
    return tasks_df.sort_values(by='priority', ascending=False)['description'].values

# Example usage
if __name__ == "__main__":
    model, vectorizer = load_model_and_vectorizer()
    tasks_df = load_dynamic_task_file()

    # new_task_description = ""
    # tasks_df = add_task(new_task_description, tasks_df, model, vectorizer)
    # tasks_df = remove_task("Implement feature XYZ", tasks_df)
    # print("List of tasks:")
    # for task in list_tasks(tasks_df):
    #     print(task)


    while True:
        print("Enter 1 to add a task")
        print("Enter 2 to remove a task")
        print("Enter 3 to list all tasks")
        print("Enter 4 to exit")
        choice = int(input())
        if choice == 1:
            new_task_description = input("Enter the task description: ")
            tasks_df = add_task(new_task_description, tasks_df, model, vectorizer)
        elif choice == 2:
            task_description = input("Enter the task description: ")
            tasks_df = remove_task(task_description, tasks_df)
        elif choice == 3:
            print("List of tasks:")
            for task in list_tasks(tasks_df):
                print(task)
        elif choice == 4:
            break
        else:
            print("Invalid choice")

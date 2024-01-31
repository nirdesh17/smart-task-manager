# Smart Task Manager

## Overview
This project aims to create a task management system that intelligently assigns priorities to tasks using a machine learning model. The system is designed to enhance task organization and streamline task management for users.

## Setup and Installation
1. **Dependencies**:
- Ensure you have Python installed 
- Install required libraries using `pip install`
2. **Training the Model**:
- Run `train.py` to load the dataset, preprocess data, train the model, and save the trained model and vectorizer.
3. **Task Management System:**
- Use `task_management.py` to interact with the task management system.
- Ensure `trained_model.pkl` and `tfidf_vectorizer.pkl` are present in the project directory.

## Usage
**Adding a Task**:
- Run the task management script.
- Choose option 1 to add a task.
- Enter the task description when prompted.
**Removing a Task**:
- Choose option 2 to remove a task.
- Enter the task description when prompted.
**Listing Tasks**:
- Choose option 3 to list all tasks in priority order (high to low).
**Exiting the System**:
- Choose option 4 to exit the system.

## Important Note
- Ensure `trained_model.pkl` and `tfidf_vectorizer.pkl` are present for the task management system to function properly.
## Future Improvements
- Potential enhancements include a more sophisticated user interface, improved input validation, and additional features like task deadlines and reminders.


## Contributing
Feel free to contribute or report issues!

### Connect me:
[Linkedin](https://www.linkedin.com/in/nirdesh-devadiya-55b408209)
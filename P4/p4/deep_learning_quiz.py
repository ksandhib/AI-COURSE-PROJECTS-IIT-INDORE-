# Deep Learning MCQ Assignment

questions = [
    {
        "question": "1. Which industries commonly use Deep Learning for image recognition applications?",
        "options": ["a. Healthcare", "b. Manufacturing", "c. Retail", "d. All of the above"],
        "answer": "d. All of the above"
    },
    {
        "question": "2. What distinguishes Deep Learning from traditional Machine Learning?",
        "options": ["a. Depth of Neural Networks", "b. Type of Algorithms used", "c. Machine Learning is an evolution of Deep Learning", "d. Volume of data processed"],
        "answer": "a. Depth of Neural Networks, d. Volume of data processed"
    },
    {
        "question": "3. Which metrics are commonly used to evaluate the performance of a classification model?",
        "options": ["a. Mean squared error", "b. Root mean squared error", "c. F1 score", "d. R-squared"],
        "answer": "c. F1 score"
    },
    {
        "question": "4. Which of the following are the key components of Deep Learning (DL)?",
        "options": ["a. Decision Trees", "b. Neural Networks", "c. Support Vector Machines", "d. K-Means Clustering"],
        "answer": "b. Neural Networks"
    },
    {
        "question": "5. What are the main challenges in training deep neural networks?",
        "options": ["a. Lack of labeled data", "b. Overfitting", "c. Underfitting", "d. None of the above"],
        "answer": "a. Lack of labeled data, b. Overfitting, c. Underfitting"
    },
    {
        "question": "6. Which technology is a subset of AI and focuses on training models to learn from data?",
        "options": ["a. Artificial Intelligence", "b. Machine Learning", "c. Deep Learning", "d. Data Science"],
        "answer": "b. Machine Learning"
    },
    {
        "question": "7. In the context of Deep Learning, what is a neural network?",
        "options": ["a. A network of interconnected computers", "b. A network of interconnected neurons", "c. A network of interconnected databases", "d. A network of interconnected algorithms"],
        "answer": "b. A network of interconnected neurons"
    },
    {
        "question": "8. What is the role of hyperparameter tuning in the model evaluation phase?",
        "options": ["a. To adjust the learning rate", "b. To optimize model parameters for better performance", "c. To select features", "d. To preprocess the data"],
        "answer": "b. To optimize model parameters for better performance"
    },
    {
        "question": "9. What is the primary goal during the project initiation phase of a Deep Learning project?",
        "options": ["a. Model training", "b. Define the problem and scope", "c. Hyperparameter tuning", "d. Data collection"],
        "answer": "b. Define the problem and scope"
    },
    {
        "question": "10. Why is continuous monitoring important in the life cycle of a deployed Deep Learning model?",
        "options": ["a. To retrain the model from scratch", "b. To ensure the model adapts to changing data patterns", "c. To deploy a new model version daily", "d. To skip the model evaluation phase"],
        "answer": "b. To ensure the model adapts to changing data patterns"
    }
]

print("Deep Learning MCQ Assignment - Correct Answers\n")

for q in questions:
    print(q["question"])
    for opt in q["options"]:
        print(opt)
    print(f"Correct Answer: {q['answer']}\n")
